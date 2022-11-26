from pytorch_lightning import LightningModule
import torch.nn.functional as F
from argparse import ArgumentParser, Namespace
from torch import Tensor
from torch.nn import Linear
from torch.optim import AdamW, Optimizer, RAdam
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_scheduler, PreTrainedModel, ViTConfig, ViTForImageClassification


class ImageClassificationNet(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Classification Model")
        parser.add_argument(
            "--optimizer",
            type=str,
            default="AdamW",
            choices=["AdamW", "RAdam"],
            help="The optimizer to use to train the model.",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-2,
            help="The optimizer's weight decay.",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=5e-5,
            help="The initial learning rate for the model.",
        )
        return parent_parser

    def __init__(
            self,
            model: PreTrainedModel,
            num_train_steps: int,
            optimizer: str = "AdamW",
            weight_decay: float = 1e-2,
            lr: float = 5e-5,
    ):
        """A PyTorch Lightning Module for a HuggingFace model used for image classification.
        Args:
            model (PreTrainedModel): a pretrained model for image classification
            num_train_steps (int): number of training steps
            optimizer (str): optimizer to use
            weight_decay (float): weight decay for optimizer
            lr (float): the learning rate used for training
        """
        super().__init__()

        # Save the hyperparameters and the model
        self.save_hyperparameters(ignore=["model"])
        self.model = model

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
            weight_decay=self.hparams.weight_decay,
            lr=self.hparams.lr,
        )
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.hparams.num_train_steps,
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


def set_clf_head(base: PreTrainedModel, num_classes: int):
    """Set the classification head of the model in case of an output mismatch.
    Args:
        base (PreTrainedModel): the model to modify
        num_classes (int): the number of classes to use for the output layer
    """
    if base.classifier.out_features != num_classes:
        in_features = base.classifier.in_features
        base.classifier = Linear(in_features, num_classes)


def model_factory(args: Namespace, own_config: bool = False) -> PreTrainedModel:
    """A factory method for creating a HuggingFace model based on the command line args.
    Args:
        args (Namespace): the argparse Namespace object
        own_config (bool): whether to create our own model config instead of a pretrained one;
            this is recommended when the model was pre-trained on another task with a different
            amount of classes for its classifier head
    Returns:
        a PreTrainedModel instance
    """
    if args.base_model == "ViT":
        # Create a new Vision Transformer
        config_class = ViTConfig
        base_class = ViTForImageClassification
    else:
        raise Exception(f"Unknown base model: {args.base_model}")

    # Get the model config
    model_cfg_args = {
        "num_channels": 3,
        "num_labels": 2,
    }
    if not own_config and args.from_pretrained:
        # Create a model from a pretrained model
        base = base_class.from_pretrained(args.from_pretrained)
        # Set the classifier head if needed
        set_clf_head(base, model_cfg_args["num_labels"])
    else:
        # Create a model based on the config
        config = config_class(**model_cfg_args)
        base = base_class(config)

    return base
