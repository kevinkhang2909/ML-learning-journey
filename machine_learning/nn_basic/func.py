import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm


def check_accuracy(loader: DataLoader, model: nn.Module, device: str) -> float:
    """
    Check accuracy of our trained model
    Parameters:
        loader: DataLoader - A loader for the dataset you want to check accuracy on
        model: nn.Module - The model you want to check accuracy on
        device: string
    Returns:
        acc: float - The accuracy of the model on the dataset given by the loader
    """

    num_correct = 0
    num_samples = 0
    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        for x, y in loader:

            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)

            # Forward
            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


def load_data(batch_size):
    root = 'dataset/'
    train_dataset = datasets.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def train_batch(train_loader, num_epochs, device, model, optimizer, scheduler):
    for epoch in range(num_epochs):
        loop = tqdm(train_loader)
        losses = []
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)

            data = data.reshape(data.shape[0], -1)

            # Forward
            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.item())

            # Backward
            optimizer.zero_grad()  # optimizer.zero_grad() if no mixed_precision
            scaler.scale(loss).backward()  # loss.backward() if no mixed_precision

            # Gradient descent or adam step
            scaler.step(optimizer)  # optimizer.step() if no mixed_precision
            scaler.update()

            # Verbose (Optional)
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=torch.rand(1).item(), acc=torch.rand(1).item())

        # After each epoch do scheduler.step, note in this scheduler we need to send
        # in loss for that epoch! This can also be set using validation loss, and also
        # in the forward loop we can do on our batch but then we might need to modify
        # the patience parameter
        # (Optional)
        mean_loss = sum(losses) / len(losses)
        mean_loss = round(mean_loss, 2)  # we should see difference in loss at 2 decimals
        scheduler.step(mean_loss)
        print(f"Average loss for epoch {epoch} was {mean_loss}")

        return model

def report(train_loader, test_loader, model, device):
    print(f'Accuracy on training set: {check_accuracy(train_loader, model, device) * 100:.2f}')
    print(f'Accuracy on test set: {check_accuracy(test_loader, model, device) * 100:.2f}')


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss()
