from pathlib import Path
import pandas as pd
from func import Regression
from lightning import Trainer, LightningModule, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint


path = Path.home() / 'OneDrive - Seagroup/ai/time_series/bike_sharing_daily.csv'
df = pd.read_csv(path)

onehot_fields = ['season', 'mnth', 'weekday', 'weathersit']
for field in onehot_fields:
    dummies = pd.get_dummies(df[field], prefix=field, drop_first=False)
    df = pd.concat([df, dummies], axis=1)
df = df.drop(onehot_fields, axis=1)

continuous_fields = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}
for field in continuous_fields:
    mean, std = df[field].mean(), df[field].std()
    scaled_features[field] = [mean, std]
    df.loc[:, field] = (df[field] - mean)/std

df_backup = df.copy()
fields_to_drop = ['instant', 'dteday', 'atemp', 'workingday']
df.drop(fields_to_drop, axis=1, inplace=True)

# Split of 60 days of data from the end of the df for validation
validation_data = df[-60:]
df = df[:-60]

# Split of 21 days of data from the end of the df for testing
test_data = df[-21:]
df = df[:-21]

# The remaining (earlier) data will be used for training
train_data = df.copy()

target_fields = ['cnt', 'casual', 'registered']

train_features, train_targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
validation_features, validation_targets = validation_data.drop(target_fields, axis=1), validation_data[target_fields]

seed_everything(42)
l_rate = 0.2
model = Regression(train_features, train_targets, validation_features, validation_targets, test_features, test_targets)
checkpoint_callback = ModelCheckpoint(dirpath="/", save_top_k=2, monitor="val_loss")
trainer = Trainer(max_epochs=50,
                  callbacks=[checkpoint_callback],
                  enable_progress_bar=True,
                  log_every_n_steps=50,
                  accelerator='cpu')
trainer.fit(model)
