import torch
from torch import nn
from typing import Dict, Optional, Any
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def make_conv_net():
    return nn.Sequential(
        nn.Conv2d(
            in_channels=3, out_channels=8, stride=1, kernel_size=(3, 3), padding=1
        ),
        nn.ReLu(),
        nn.Conv2d(
            in_channels=8, out_channels=32, kernel_size=(3, 3), padding=1, stride=1
        ),
        nn.ReLu(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1
        ),
        nn.ReLu(),
        nn.Dropout2d(p=0.5),
        nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1
        ),
        nn.ReLu(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
        nn.ReLu(),
        nn.Dropout2d(p=0.5),
        nn.Flatten(),
        nn.Linear(in_features=6 * 6 * 256, out_features=256),
        nn.ReLu(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLu(),
        nn.Linear(in_features=128, out_features=64),
        nn.ReLu(),
        nn.Linear(in_features=64, out_features=10),
    )


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


def train_net(
    epochs,
    net,
    trainset,
    lr=0.001,
    plot=False,
    preproc=False,
    dataset_name="mnist",
    testset=None,
):
    """
    Trains inputted net using provided trainset.
    """

    # log on wandb
    wandb.init(
        project="boundary_learning",
        config={
            "lr": lr,
            "dataset_name": dataset_name,
            "epochs": epochs,
        },
    )

    if preproc:
        preproc_data = []
        for batch in trainset:
            X, y = batch
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            preproc_data.append([X, y])
        trainset = preproc_data

    criterion = nn.CrossEntropyLoss()
    if dataset_name == "mnist":
        optimizer = torch.optim.Adam(net.parameters(), lr)
    elif dataset_name == "cifar10":
        optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    else:
        raise ValueError(f"Dataset {dataset_name} not one of (mnist, cifar10)")

    losses = []
    epoch_losses = []

    net.to(DEVICE)
    for epoch in tqdm(range(epochs)):
        net.train()
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

        log_data = {"train_loss": epoch_loss}
        if testset is not None:
            test_loss, test_acc = test_net(net, testset)
            log_data["test_loss"] = test_loss
            log_data["test_acc"] = test_acc

        wandb.log(log_data)

    if plot:
        plt.plot([i for i in range(len(losses[10:]))], losses[10:])
        plt.title("Training Loss")
        plt.xlabel("Batch")
        plt.show()

        plt.plot([i for i in range(len(epoch_losses))], epoch_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.show()

    net.eval()

    return epoch_losses
