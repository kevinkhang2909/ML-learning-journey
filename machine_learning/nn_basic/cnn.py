"""
How to code a convolutional neural network (CNN)
Programmed by Kevin
* 2022-12-25: Initial coding
"""

import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from func import report, load_data, train_batch


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channels = 1
num_classes = 10
learning_rate = 3e-4
batch_size = 64
num_epochs = 3

# Load Data
train_loader, test_loader = load_data(batch_size)

# Init model
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
# model = nn.Sequential(
#     nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1, stride=1),
#     nn.Flatten(),
#     nn.Linear(10 * 224 * 224, 10),
# )

# Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

# Train Network
model = train_batch(train_loader, num_epochs, device, model, optimizer, scheduler)

report(train_loader, test_loader, model, device)
