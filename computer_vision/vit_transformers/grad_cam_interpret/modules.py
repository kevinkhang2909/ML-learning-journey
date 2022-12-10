import torch
from torch import Tensor

import cv2
import numpy as np
from PIL import Image
from typing import List, Callable, Optional

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_lightning import LightningModule


class HuggingfaceWrapper(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).logits


def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_image: Image,
                          input_tensor: torch.nn.Module,
                          method: Callable = GradCAM):
    """
    Helper function to run GradCAM on an image and create a visualization.
    If several targets are passed in targets_for_gradcam,
    a visualization for each of them will be created.
    """
    # Method
    cam = method(model=HuggingfaceWrapper(model),
                 target_layers=[target_layer],
                 reshape_transform=reshape_transform)

    # Replicate the tensor for each of the categories we want to create Grad-CAM for:
    repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)
    batch_results = cam(input_tensor=repeated_tensor, targets=targets_for_gradcam)

    # Results
    results = []
    for grayscale_cam in batch_results:
        visualization = show_cam_on_image(np.float32(input_image) / 255, grayscale_cam, use_rgb=True)
        # Make it weight less in the notebook:
        visualization = cv2.resize(visualization, (visualization.shape[1] // 2, visualization.shape[0] // 2))
        results.append(visualization)
    return np.hstack(results)


def reshape_vit_huggingface(x):
    """
    Reshaping to features with the format: batch x features x height x width
    Transformers will sometimes have an internal shape that looks like this:
    (batch=10 x (tokens=145) x (features=384).
    The 145 tokens mean 1 CLS token + 144 spatial tokens.
    These 144 spatial tokens actually represent a 12x12 2D image.
    """
    # Remove the CLS token
    activations = x[:, 1:, :]
    image_2d_shape = int(np.sqrt(activations.shape[1]))
    # Reshape to a 12 x 12 spatial image:
    activations = activations.view(activations.shape[0], image_2d_shape, image_2d_shape, activations.shape[2])
    # Transpose the features to be in the second coordinate:
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations
