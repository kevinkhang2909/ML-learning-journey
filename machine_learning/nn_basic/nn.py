"""
How to code a fully connected neural network
Programmed by Kevin
* 2022-12-25: Initial coding
"""

import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from func import load_data, report, train_batch


class NN(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        """
        Define the layers of the network with two fully connected layers
        Parameters:
            input_size: the size of the input (784 - 28x28)
            num_classes: the number of classes to predict (10)
        """
        super(NN, self).__init__()
        # First linear layer take input_size (784) nodes to 50
        self.fc1 = nn.Linear(input_size, 50)
        # Second linear layer takes 50 to the num_classes
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above
        Adding a ReLU activation function in between
        Parameters:
            x: mnist images
        Returns:
            out: the output of the network
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# Load Data
train_loader, test_loader = load_data(batch_size)

# Init model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

# Train Network
model = train_batch(train_loader, num_epochs, device, model, optimizer, scheduler)

report(train_loader, test_loader, model, device)
