import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from transformers import ViTForImageClassification

from datamodules import ImageDataModule
from models_vit_v2 import ImageClassificationNet

# Seed
pl.seed_everything(42)


class CONFIG:
    dataset = 'dog_cat'
    data_dir = str(Path.home() / f'Desktop/{dataset}')
    model_name = 'google/vit-base-patch16-224-in21k'
    checkpoint = None
    num_epochs = 4
    optimizer = 'AdamW'  # choices: RAdam
    lr = 5e-5
    num_class = 2


# Datamodule
dm = ImageDataModule(model_name=CONFIG.model_name, data_dir=CONFIG.data_dir)
train_ds, val_ds, test_ds = dm.setup()
id2label = {idx: label for idx, label in enumerate(train_ds.features['label'].names)}
label2id = {label: idx for idx, label in id2label.items()}

# Model base
model_base = ViTForImageClassification.from_pretrained(CONFIG.model_name,
                                                       num_labels=CONFIG.num_class,
                                                       id2label=id2label,
                                                       label2id=label2id)

if CONFIG.checkpoint:
    # Load the model from the specified checkpoint
    model = ImageClassificationNet.load_from_checkpoint(CONFIG.checkpoint, model=model_base)
else:
    # Create a new instance of the classification model
    model = ImageClassificationNet(
        model=model_base,
        num_train_steps=CONFIG.num_epochs * len(dm.train_dataloader()),
        optimizer=CONFIG.optimizer,
        lr=CONFIG.lr,
    )

# Create wandb logger
wandb_logger = WandbLogger(
    name=f"{CONFIG.dataset}_training ({CONFIG.model_name})",
    project=f"Classification-{CONFIG.dataset}",
)

# Create checkpoint callback
ckpt_cb = ModelCheckpoint(dirpath=f"checkpoints/{wandb_logger.version}")
# Create early stopping callback
es_cb = EarlyStopping(monitor="val_acc", mode="max", patience=5)

# Create trainer
trainer = pl.Trainer(
    accelerator="auto",
    callbacks=[ckpt_cb, es_cb],
    logger=wandb_logger,
    max_epochs=CONFIG.num_epochs,
)

trainer_args = {}
if CONFIG.checkpoint:
    # Resume trainer from checkpoint
    trainer_args["ckpt_path"] = CONFIG.checkpoint

# Train the model
trainer.fit(model, dm, **trainer_args)
