import cv2
import itertools
import albumentations as A
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import DistilBertModel, ResNetModel
from lightning import LightningModule


class Config:
    # pretrain_image = 'microsoft/resnet-50'
    pretrain_image = 'google/vit-base-patch16-224-in21k'
    # pretrain_text = 'distilbert-base-uncased'
    pretrain_text = 'sentence-transformers/all-MiniLM-L6-v2'
    device = 'cuda'


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=Config.pretrain_image):
        super().__init__()
        self.model = ResNetModel.from_pretrained(pretrained)

    def forward(self, input_ids):
        output = self.model(input_ids)
        output = output.pooler_output.squeeze()
        return output


class TextEncoder(nn.Module):
    def __init__(self, pretrained=Config.pretrain_text):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(pretrained)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, 0, :]


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)


class CLIPDualEncoderModel(LightningModule):
    def __init__(self,
                 image_pretrained: str,
                 text_pretrained: str,
                 image_embedding_dims: int = 2048,
                 text_embedding_dims: int = 384,
                 projection_dims: int = 256,
                 dropout: float = 0.0,
                 temperature: float = 1.0,
                 weight_decay: float = 0.1,
                 head_lr: float = 1e-2,
                 image_encoder_lr: float = 1e-2,
                 text_encoder_lr: float = 1e-2,
                 lr_scheduler_patience: int = 1,
                 lr_scheduler_factor: float = 0.8,
                 batch_size: int = 64,
                 max_epochs: int = 3,
                 *args,
                 **kwargs,
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder = ImageEncoder(pretrained=image_pretrained)
        self.text_encoder = TextEncoder(pretrained=text_pretrained)
        self.image_projection = ProjectionHead(
            embedding_dim=image_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.image_encoder_lr = image_encoder_lr
        self.text_encoder_lr = text_encoder_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.save_hyperparameters()

    def _compute_losses(self, image_embeddings, text_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        return (images_loss + texts_loss) / 2.0

    def forward(self, inputs):
        image_features = self.image_encoder(inputs["image"])
        text_features = self.text_encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return image_embeddings, text_embeddings

    def configure_optimizers(self):
        parameters = [
            {"params": self.image_encoder.parameters(), "lr": self.image_encoder_lr},
            {"params": self.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {"params": itertools.chain(self.image_projection.parameters(), self.text_projection.parameters()),
             "lr": self.head_lr, "weight_decay": self.weight_decay},
        ]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-6)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     patience=self.lr_scheduler_patience,
        #     factor=self.lr_scheduler_factor,
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        train_loss = self.all_gather(loss)
        self.log("train_loss", train_loss.mean(), on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        val_loss = self.all_gather(loss)
        self.log("val_loss", val_loss.mean(), on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss


def get_transform():
    trf = A.Compose([
        A.Resize(224, 224, always_apply=True),
        A.Normalize(max_pixel_value=255.0, always_apply=True)
        ]
    )
    return trf


class FlickrDataset(Dataset):
    def __init__(self, image_filenames, captions, tokenizer):
        self.image_filenames = image_filenames
        self.captions = captions
        self.encoded_captions = tokenizer(captions, padding=True, truncation=True, max_length=32)
        self.transforms = get_transform()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encoded_captions.items()}
        image = cv2.imread(self.image_filenames[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)
