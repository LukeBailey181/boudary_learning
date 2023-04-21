import torch
from torch import nn
from typing import Dict, Optional, Any
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def make_net(
    input_dim: int,
    num_classes: int,
    hidden_units: int,
    hidden_layers: int,
    dropout_layer: Optional[Any] = None,
    dropout_kwargs: Dict[str, Any] = {},
) -> nn.Module:
    """Helper function for making NNs"""

    input = [
        nn.Linear(input_dim, hidden_units),
        nn.ReLU(),
    ]
    if dropout_layer is not None:
        input.append(dropout_layer(**dropout_kwargs))

    hidden = []
    for _ in range(hidden_layers):
        hidden.append(nn.Linear(hidden_units, hidden_units))
        hidden.append(nn.ReLU())
        if dropout_layer is not None:
            hidden.append(dropout_layer(**dropout_kwargs))

    output = [nn.Linear(hidden_units, num_classes)]

    return nn.Sequential(*input, *hidden, *output)

def make_dropoout_net(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 100,
    hidden_layers: int = 2,
    p: float = 0.5,
) -> nn.Module:
    """Return a NN that uses standard pytorch dropout"""

    return make_net(
        input_dim,
        num_classes,
        hidden_units,
        hidden_layers,
        dropout_layer=nn.Dropout,
        dropout_kwargs={"p": p},
    )

def make_standard_net(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 100,
    hidden_layers: int = 2,
) -> nn.Module:
    """Return a NN without dropout"""

    return make_net(
        input_dim, num_classes, hidden_units, hidden_layers, dropout_layer=None
    )


def test_net(net, dataset):
    """
    Evaulates inputted net on inputted dataset
    """

    criterion = nn.CrossEntropyLoss()
    net.to(DEVICE)
    net.eval()
    total_loss = total_correct = total_examples = 0
    with torch.no_grad():
        for data in dataset:

            X, y = data
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            output = net(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == y).sum().item()
            total_examples += len(y)

    return total_loss, total_correct / total_examples


def train_net(epochs, net, trainset, lr=0.001, plot=False, preproc=False):
    """
    Trains inputted net using provided trainset.
    """
    if preproc:
        preproc_data = []
        for batch in trainset:
            X, y = batch
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            preproc_data.append([X, y])
        trainset = preproc_data

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)

    losses = []
    epoch_losses = []

    net.train()
    net.to(DEVICE)
    for epoch in tqdm(range(epochs)):

        epoch_loss = 0

        for data in trainset:
            X, y = data
            if not preproc:
                X = X.to(DEVICE)
                y = y.to(DEVICE)

            net.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss)

    if plot:
        plt.plot([i for i in range(len(losses[10:]))], losses[10:])
        plt.title("Training Loss")
        plt.xlabel("Batch")
        plt.show()

        plt.plot([i for i in range(len(epoch_losses))], epoch_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.show()

    return epoch_losses
