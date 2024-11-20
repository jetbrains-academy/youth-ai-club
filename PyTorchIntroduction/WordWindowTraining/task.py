from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


criterion = nn.BCELoss()


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, device: str = 'cpu')\
        -> float:
    model.train()  # move model into train mode

    losses = []
    for word_idx_padded, labels, lengths in train_loader:
        # move tensors into device
        word_idx_padded, labels, lengths = word_idx_padded.to(device), labels.to(device), lengths.to(device)
        """
        word_idx_padded: [B x L]
        labels: [B x L~]
        lengths: [B]
        """
        optimizer.zero_grad()
        predictions = model(word_idx_padded) # predictions: [B x L~]
        loss = criterion(predictions, labels.float())
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return sum(losses) / len(losses) # Not the best practice as last batch can be shorter than others


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> Tuple[float, float]:
    model.eval()

    losses = []
    accs = []
    for word_idx_padded, labels, lengths in test_loader:
        word_idx_padded, labels, lengths = word_idx_padded.to(device), labels.to(device), lengths.to(device)
        """
        word_idx_padded: [B x L]
        labels: [B x L~]
        lengths: [B]
        """
        predictions = model(word_idx_padded)  # predictions: [B x L~]
        loss = criterion(predictions, labels.float())

        losses.append(loss.item())
        accs.append(torch.mean(((predictions > 0.5).long() == labels).float()).item())  # calculating accuracy
    return sum(losses) / len(losses), sum(accs) / len(accs) # Not the best practice as last batch can be shorter than others


def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
          epochs: int, device: str = 'cpu', validate_every_n_epochs: int = 10)\
        -> Tuple[List[float], List[float], List[float]]:
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    train_losses, val_losses, val_accs = [], [], []

    model.to(device)
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        print(f"Train iter {epoch}: {train_loss}")

        if (epoch + 1) % validate_every_n_epochs == 0 or epoch + 1 == epochs:
            val_loss, val_acc = evaluate(model, test_loader, device)
            val_losses.append(val_losses)
            val_accs.append(val_acc)
            print(f"Validate iter {epoch}: {val_loss=}, {val_acc=}")

    return train_losses, val_losses, val_accs
