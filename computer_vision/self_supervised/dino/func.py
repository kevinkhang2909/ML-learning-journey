import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms, io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from lightning import LightningModule


# IMAGE_SIZE = 256
PATCH_SIZE = 16
ZERO_PCT = 0.1
NUM_PATCHES = (256 // PATCH_SIZE) ** 2
RGB_CHANNELS = 3
TOPK = 5
VALID_IMAGES = 5


class ImageData(Dataset):
    def __init__(self, files, image_size=256):
        self.files = files
        self.randcrop_big = transforms.RandomResizedCrop((image_size, image_size), scale=(0.5, 1.0), antialias=True)
        self.randcrop_small = transforms.RandomResizedCrop((image_size, image_size), antialias=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = io.read_image(self.files[i])
        img1 = self.randcrop_big(img)
        img2 = self.randcrop_small(img)
        if img.shape[0] == 1:
            img1 = torch.cat([img1] * 3)
            img2 = torch.cat([img2] * 3)

        return img1, img2


class CollateFn:
    def reshape(self, batch):
        patches = torch.stack(batch).unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)

        num_images = len(patches)
        patches = patches.reshape(
            num_images,
            RGB_CHANNELS,
            NUM_PATCHES,
            PATCH_SIZE,
            PATCH_SIZE
        )
        patches.transpose_(1, 2)

        return patches.reshape(num_images, NUM_PATCHES, -1) / 255.0 - 0.5

    def __call__(self, batch):
        x1, x2 = zip(*batch)
        return self.reshape(x1), self.reshape(x2)


class ImageOriginalData(Dataset):
    def __init__(self, files, image_size=256):
        self.files = files
        self.resize = transforms.Resize((image_size, image_size))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = io.read_image(self.files[i])
        if img.shape[0] == 1:
            img = torch.cat([img] * 3)
        return self.resize(img)


class CollateSingleImage(CollateFn):
    def __call__(self, batch) -> torch.FloatTensor:
        return self.reshape(batch)


class Model(nn.Module):
    def __init__(self, d_model, n_head, n_layers):
        super().__init__()
        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # positional embedding
        w_pos = torch.randn(NUM_PATCHES, d_model) / d_model ** 0.5
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
    def __init__(self, model, lr, loss_fn, valid_files):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn
        self.valid_files = valid_files

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
        self.log(name="train_loss", value=loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, images: torch.FloatTensor, *args) -> None:
        return self.model(images)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
