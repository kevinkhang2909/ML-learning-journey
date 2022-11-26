from torch.utils.data import random_split
from torchvision.datasets import MNIST
from typing import Optional
from base import ImageDataModule
from transformations import UnNest
from transformers import ConvNextFeatureExtractor, ViTFeatureExtractor, ViTForImageClassification
from argparse import ArgumentParser, Namespace


class MNISTDataModule(ImageDataModule):
    """Datamodule for the MNIST dataset."""

    def prepare_data(self):
        # Download MNIST
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Set the training and validation data
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = random_split(mnist_full, [55000, 5000])

        # Set the test data
        if stage == "test" or stage is None:
            self.test_data = MNIST(self.data_dir, train=False, transform=self.transform)


def get_configs(args: Namespace) -> tuple[dict, dict]:
    """Get the model and feature extractor configs from the command line args.
    Args:
        args (Namespace): the argparse Namespace object
    Returns:
         a tuple containing the model and feature extractor configs
    """
    # We upsample the MNIST images to 112x112, with 1 channel (grayscale)
    # and 10 classes (0-9). We normalize the image to have a mean of 0.5
    # and a standard deviation of ±0.5.
    model_cfg_args = {
        "image_size": 112,
        "num_channels": 1,
        "num_labels": 10,
    }
    fe_cfg_args = {"image_mean": [0.5], "image_std": [0.5], "size": model_cfg_args["image_size"],
                   "return_tensors": "pt"}

    # Set the feature extractor's size attribute to  be the same as the model's image size
    # Set the tensors' return type to PyTorch tensors
    return model_cfg_args, fe_cfg_args


def datamodule_factory(args: Namespace) -> ImageDataModule:
    """A factory method for creating a datamodule based on the command line args.
    Args:
        args (Namespace): the argparse Namespace object
    Returns:
        an ImageDataModule instance
    """
    # Get the model and feature extractor configs
    model_cfg_args, fe_cfg_args = get_configs(args)

    # Set the feature extractor class based on the provided base model name
    if args.base_model == "ViT":
        fe_class = ViTFeatureExtractor
    elif args.base_model == "ConvNeXt":
        fe_class = ConvNextFeatureExtractor
    else:
        raise Exception(f"Unknown base model: {args.base_model}")

    # Create the feature extractor instance
    if args.from_pretrained:
        feature_extractor = fe_class.from_pretrained(
            args.from_pretrained, **fe_cfg_args
        )
    else:
        feature_extractor = fe_class(**fe_cfg_args)

    # Un-nest the feature extractor's output
    feature_extractor = UnNest(feature_extractor)

    # Define the datamodule's configuration
    dm_cfg = {
        "feature_extractor": feature_extractor,
        "batch_size": args.batch_size,
        "add_noise": args.add_noise,
        "add_rotation": args.add_rotation,
        "add_blur": args.add_blur,
        "num_workers": args.num_workers,
    }

    # Determine the dataset class based on the provided dataset name
    if args.dataset == "MNIST":
        dm_class = MNISTDataModule
    else:
        raise Exception(f"Unknown dataset: {args.dataset}")

    return dm_class(**dm_cfg)


parser = ArgumentParser()

# Trainer
parser.add_argument(
    "--enable_progress_bar",
    action="store_true",
    help="Whether to enable the progress bar (NOT recommended when logging to file).",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=5,
    help="Number of epochs to train.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123,
    help="Random seed for reproducibility.",
)

# Logging
parser.add_argument(
    "--sample_images",
    type=int,
    default=8,
    help="Number of images to sample for the mask callback.",
)
parser.add_argument(
    "--log_every_n_steps",
    type=int,
    default=200,
    help="Number of steps between logging media & checkpoints.",
)

# Base (classification) model
parser.add_argument(
    "--base_model",
    type=str,
    default="ViT",
    choices=["ViT"],
    help="Base model architecture to train.",
)
parser.add_argument(
    "--from_pretrained",
    type=str,
    default="tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
    help="The name of the pretrained HF model to load.",
)

# Interpretation model
# ImageInterpretationNet.add_model_specific_args(parser)

# Datamodule
ImageDataModule.add_model_specific_args(parser)
parser.add_argument(
    "--dataset",
    type=str,
    default="MNIST",
    choices=["MNIST", "CIFAR10", "CIFAR10_QA", "toy"],
    help="The dataset to use.",
)
args = parser.parse_args()

# Load pre-trained Transformer
model_checkpoint = 'google/vit-base-patch16-224-in21k'
model = ViTForImageClassification.from_pretrained(model_checkpoint)

# # Load datamodule
# dm = datamodule_factory(args)
#
# # Setup datamodule to sample images for the mask callback
# dm.prepare_data()
# dm.setup("fit")

# n_panels = 2
# images_per_panel = args.sample_images
#
# # Sample images
# sample_images = []
# iter_loader = iter(dm.val_dataloader())
# for panel in range(n_panels):
#     X, Y = next(iter_loader)
#     sample_images += [(X[:images_per_panel], Y[:images_per_panel])]