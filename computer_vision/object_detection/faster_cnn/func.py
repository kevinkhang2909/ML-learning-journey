import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.patches as patches
from PIL import Image


class Config:
    classes = [str(i) for i in range(15)] + ['__background__']


def add_bounding_box_label(box, label, ax):
    bb = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(bb)
    ax.text(box[0], (box[1]-10), label, color='blue', fontsize=18, weight='bold')
    return ax


def show(dataset, num_image: int = 5, transform=None):
    fig, axes = plt.subplots(1, num_image, figsize=(20, 5))
    axes = axes.flatten()

    for i in range(num_image):
        image, bounding_boxes = dataset[i]
        bboxes = [_['bbox'] for _ in bounding_boxes]
        labels = [str(_['category_id']) for _ in bounding_boxes]

        if transform:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = transform(image=image, bboxes=bboxes, class_labels=labels)
            for box, label in zip(image['bboxes'], image['class_labels']):
                axes[i] = add_bounding_box_label(box, label, axes[i])
            axes[i].imshow(Image.fromarray(image['image']))
        else:
            for box, label in zip(bboxes, labels):
                axes[i] = add_bounding_box_label(box, label, axes[i])
            axes[i].imshow(image)
    fig.tight_layout()


def get_train_transform():
    return A.Compose(
        [
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'])
    )


def get_valid_transform():
    return A.Compose([ToTensorV2(p=1.0)], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
