from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, LightningDataModule
from transformers import ViTForImageClassification, AdamW, ViTFeatureExtractor
from datasets import load_dataset
from torchvision import transforms


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "label": labels}


def transform_img(feature_extractor, mode='train'):
    if mode == 'train':
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(feature_extractor.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(feature_extractor.size),
                transforms.CenterCrop(feature_extractor.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ]
        )


class DataModule(LightningDataModule):
    def __init__(self, data_dir, model_name):
        super().__init__()
        self.test_ds = None
        self.val_ds = None
        self.train_ds = None
        self.data_dir = data_dir
        self.model_name = model_name
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    def transforms_process_train(self, examples):
        examples['pixel_values'] = [transform_img(self.feature_extractor, mode='train')(image.convert("RGB"))
                                    for image in examples['image']]
        return examples

    def transforms_process_test(self, examples):
        examples['pixel_values'] = [transform_img(self.feature_extractor, mode='test')(image.convert("RGB"))
                                    for image in examples['image']]
        return examples

    def setup(self, stage=None):
        """ called on each GPU separately - stage defines if we are at fit or test step """
        dataset_train = load_dataset("imagefolder", data_dir=self.data_dir, split='train')
        splits = dataset_train.train_test_split(test_size=0.2)
        # train & valid
        dataset_train_valid = splits['train'].train_test_split(test_size=0.1)
        self.train_ds = dataset_train_valid['train']
        self.val_ds = dataset_train_valid['test']
        # test
        self.test_ds = splits['test']
        # transform
        self.train_ds.set_transform(self.transforms_process_train)
        self.val_ds.set_transform(self.transforms_process_test)
        self.test_ds.set_transform(self.transforms_process_test)
        return self.train_ds, self.val_ds, self.test_ds


class ViTLightning(LightningModule):
    def __init__(self, model_name, train_ds, val_ds, test_ds, lr=5e-5, batch_size=256, num_workers=0):
        super().__init__()
        self.test_ds = test_ds
        self.val_ds = val_ds
        self.train_ds = train_ds
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.id2label = {idx: label for idx, label in enumerate(self.train_ds.features['label'].names)}
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        self.num_class = len(self.id2label)
        self.vit = ViTForImageClassification.from_pretrained(self.model_name, num_labels=self.num_class,
                                                             id2label=self.id2label, label2id=self.label2id)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['label']
        logits = self(pixel_values)

        loss = nn.CrossEntropyLoss()(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", accuracy, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=collate_fn)
