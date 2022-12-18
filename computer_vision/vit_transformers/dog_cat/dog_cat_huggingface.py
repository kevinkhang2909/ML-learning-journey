from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import default_collate
from evaluate import load
from pathlib import Path
import numpy as np


def compute_metrics(p):
    metric = load("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def transforms(example_batch):
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs


# Input
model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

path_train = Path('/content/drive/MyDrive/Colab Notebooks/data/dogs-vs-cats')
dataset_train = load_dataset("imagefolder", data_dir=str(path_train), split='train')
splits = dataset_train.train_test_split(test_size=0.2)
dataset_test_valid = splits['test'].train_test_split(test_size=0.5)

# Set the train and validation data
train_data, val_data = splits['train'], dataset_test_valid['train']
train_data.set_transform(transforms)
val_data.set_transform(transforms)

# Set the test data
test_data = dataset_test_valid['test']
test_data.set_transform(transforms)

# Model
labels = {'cat': 0, 'dog': 1}
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

# Train
training_args = TrainingArguments(
    output_dir="./vit_dog_cat",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=3,
    fp16=True,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=default_collate,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
