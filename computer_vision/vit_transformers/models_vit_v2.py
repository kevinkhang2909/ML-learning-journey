from pytorch_lightning import LightningModule
import torch.nn.functional as F
from argparse import ArgumentParser
from torch import Tensor
from torch.optim import AdamW, Optimizer, RAdam
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_scheduler, PreTrainedModel


class ImageClassificationNet(LightningModule):
    def __init__(
            self,
            model: PreTrainedModel,
            num_train_steps: int,
            optimizer: str = "AdamW",
            lr: float = 5e-5,
            weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.model = model
        self.num_train_steps = num_train_steps
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).logits

    def configure_optimizers(self) -> tuple[list[Optimizer], list[_LRScheduler]]:
        # Set the optimizer class based on the hyperparameter
        if self.hparams.optimizer == "AdamW":
            optim_class = AdamW
        elif self.hparams.optimizer == "RAdam":
            optim_class = RAdam
        else:
            raise Exception(f"Unknown optimizer {self.hparams.optimizer}")

        # Create the optimizer and the learning rate scheduler
        optimizer = optim_class(
            self.parameters(),
            weight_decay=self.weight_decay,
            lr=self.lr,
        )
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps,
        )

        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch: tuple[Tensor, Tensor], mode: str) -> Tensor:
        imgs, labels = batch

        preds = self.model(imgs).logits
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)

        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], _: Tensor) -> Tensor:
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], _: Tensor):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch: tuple[Tensor, Tensor], _: Tensor):
        self._calculate_loss(batch, mode="test")
