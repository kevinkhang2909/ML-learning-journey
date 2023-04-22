from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from lightning import LightningModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class Config:
    image_size = 256
    patch_size = 16
    num_patches = (256 // 16) ** 2
    rgb_channel = 3
    batch_size = 32
    num_workers = 4
    num_pixels = 256 * 3  # image_size * rgb channel


class ImageData(Dataset):
    def __init__(self, files, image_size=256):
        self.files = files
        self.randcrop_big = A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.5, 1.0)),
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
            ToTensorV2(),
        ])
        self.randcrop_small = A.Compose([
            A.RandomResizedCrop(image_size, image_size),
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = self.randcrop_big(image=img)['image']
        img2 = self.randcrop_small(image=img)['image']
        if img.shape[0] == 1:
            img1 = torch.cat([img1] * 3)
            img2 = torch.cat([img2] * 3)

        return img1, img2


class CollateFn:
    def reshape(self, batch):
        patches = torch.stack(batch).unfold(2, Config.patch_size, Config.patch_size).unfold(3, Config.patch_size, Config.patch_size)

        num_images = len(patches)
        patches = patches.reshape(num_images, Config.rgb_channel, Config.num_patches, Config.patch_size, Config.patch_size)
        patches.transpose_(1, 2)

        return patches.reshape(num_images, Config.num_patches, -1) / 255.0 - 0.5

    def __call__(self, batch):
        x1, x2 = zip(*batch)
        return self.reshape(x1), self.reshape(x2)


class ImageOriginalData(Dataset):
    def __init__(self, files, image_size=256):
        self.files = files
        self.resize = A.Compose([
            A.Resize(image_size, image_size),
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] == 1:
            img = torch.cat([img] * 3)
        return self.resize(image=img)['image']


class CollateSingleImage(CollateFn):
    def __call__(self, batch) -> torch.FloatTensor:
        return self.reshape(batch)


class Model(nn.Module):
    def __init__(self, d_model=Config.num_pixels, n_head=8, n_layers=6):
        super().__init__()
        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # positional embedding
        w_pos = torch.randn(Config.num_patches, d_model) / d_model ** 0.5
        cls_token = torch.randn(1, d_model) / d_model ** 0.5
        self.register_parameter("pos_embed", nn.Parameter(w_pos))
        self.register_parameter("cls_token", nn.Parameter(cls_token))

        # pixel projection
        self.linear = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(2 * d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size = len(x)
        position = torch.stack([self.pos_embed] * batch_size)
        x = torch.cat([x, position], dim=-1)
        pixel_proj = self.norm2(F.relu(self.linear(self.norm1(x))))
        batched_cls_token = torch.stack([self.cls_token] * batch_size)
        cls_x = torch.cat([batched_cls_token, pixel_proj], dim=1)

        cls_x.transpose_(0, 1)
        return F.normalize(self.encoder(cls_x)[0, ...], dim=-1)


class HLoss:
    def __init__(self, temperature_t: float, temperature_s: float):
        self.temperature_t = temperature_t
        self.temperature_s = temperature_s

    def __call__(self, t: torch.FloatTensor, s: torch.FloatTensor, center: torch.FloatTensor):
        t = F.softmax((t.detach() - center) / self.temperature_t, dim=1)
        log_s = F.log_softmax(s / self.temperature_s, dim=1)
        return -(t * log_s).sum(dim=1).mean()


def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    return contrastive_loss(similarity, dim=0) + contrastive_loss(similarity, dim=1)


class LightningModel(LightningModule):
    def __init__(self, model, loss_fn, valid_files, max_epochs, learning_rate, weight_decay, batch_size):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.valid_files = valid_files
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def common_step(self, batch):
        x1, x2 = batch
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16),\
                torch.backends.cuda.sdp_kernel(enable_flash=False):
            out1, out2 = self.model(x1), self.model(x2)
        similarity = out1 @ out2.T
        loss = self.loss_fn(similarity)
        return loss

    def training_step(self, batch):
        loss = self.common_step(batch)
        self.log(name="train_loss", value=loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, images: torch.FloatTensor, *args) -> None:
        return self.model(images)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        # return optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
