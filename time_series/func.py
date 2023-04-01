import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from lightning import LightningModule


l_rate = 0.2
mse_loss = nn.MSELoss(reduction='mean')


class Regression(LightningModule):
    def __init__(self, train_features, train_targets, valid_features, valid_targets, test_features, test_targets):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(train_features.shape[1], 10)
        self.fc2 = nn.Linear(10, 1)
        self.train_features = train_features
        self.train_targets = train_targets
        self.valid_features = valid_features
        self.valid_targets = valid_targets
        self.test_features = test_features
        self.test_targets = test_targets

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_dataloader(self):
        train_dataset = TensorDataset(torch.tensor(self.train_features.values).float(), torch.tensor(self.train_targets[['cnt']].values).float())
        train_loader = DataLoader(dataset=train_dataset, batch_size=128)
        return train_loader

    def val_dataloader(self):
        validation_dataset = TensorDataset(torch.tensor(self.valid_features.values).float(), torch.tensor(self.valid_targets[['cnt']].values).float())
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=128)
        return validation_loader

    def test_dataloader(self):
        test_dataset = TensorDataset(torch.tensor(self.test_features.values).float(), torch.tensor(self.test_targets[['cnt']].values).float())
        test_loader = DataLoader(dataset=test_dataset, batch_size=128)
        return test_loader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=l_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = mse_loss(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # logs = {'loss': loss}
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = mse_loss(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = mse_loss(logits, y)
        correct = torch.sum(logits == y.data)

        # predictions_pred.append(logits)
        # predictions_actual.append(y.data)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_correct': correct, 'logits': logits}
