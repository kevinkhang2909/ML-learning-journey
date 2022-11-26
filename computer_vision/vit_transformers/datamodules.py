from pytorch_lightning import LightningDataModule
from torchvision import transforms
from typing import Optional
from datasets import load_dataset
from transformers import ViTFeatureExtractor


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


class ImageDataModule(LightningDataModule):
    def __init__(self, model_name, data_dir: str = "data/"):
        super().__init__()
        self.data_dir = data_dir
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    def transforms_process_train(self, examples):
        examples['pixel_values'] = [transform_img(self.feature_extractor, mode='train')(image.convert("RGB"))
                                    for image in examples['image']]
        return examples

    def transforms_process_test(self, examples):
        examples['pixel_values'] = [transform_img(self.feature_extractor, mode='test')(image.convert("RGB"))
                                    for image in examples['image']]
        return examples

    def prepare_data(self):
        # No need to download anything for the toy task
        pass

    def setup(self, stage=None):
        dataset_train = load_dataset("imagefolder", data_dir=self.data_dir, split='train')
        splits = dataset_train.train_test_split(test_size=0.2)
        dataset_test_valid = splits['test'].train_test_split(test_size=0.5)

        # Set the train and validation data
        self.train_data, self.val_data = splits['train'], dataset_test_valid['train']
        self.train_data.set_transform(self.transforms_process_train)
        self.val_data.set_transform(self.transforms_process_test)

        # Set the test data
        self.test_data = dataset_test_valid['test']
        self.test_data.set_transform(self.transforms_process_test)

        return self.train_data, self.val_data, self.test_data
