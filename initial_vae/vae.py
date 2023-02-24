from __future__ import print_function
import math
import torch
import torch.distributions as dist
from torch.nn import functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


import os
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"



class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc31 = nn.Linear(400, latent_dim)
        self.fc32 = nn.Linear(400, latent_dim)
        self.fc4 = nn.Linear(latent_dim, 400)
        self.fc5 = nn.Linear(400, 400)
        self.fc6 = nn.Linear(400, 784)

    def encode(self, x):
        net = x
        net = F.relu(self.fc1(net))
        net = F.relu(self.fc2(net))
        return self.fc31(net), self.fc32(net).exp()

    def decode(self, z):
        net = z
        net = F.relu(self.fc4(net))
        net = F.relu(self.fc5(net))
        return self.fc6(net)

    def forward(self, x, k=1):
        x = x.view(-1, 784)
        qz_x = dist.Normal(*self.encode(x))
        z = qz_x.rsample(torch.Size([k]))
        px_z = dist.Bernoulli(logits=self.decode(z), validate_args=False)
        return qz_x, px_z, z


def compute_elbo(x, qz_x, px_z, z):
    x = x.view(-1, 784)
    lpx_z = px_z.log_prob(x).sum(-1)
    lpz = dist.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
    lqz_x = qz_x.log_prob(z).sum(-1)

    lw = lpz + lpx_z - lqz_x
    return (torch.logsumexp(lw, 0) - math.log(lw.size(0))).mean(0)


def compute_elbo_dreg(x, qz_x, px_z, z):
    x = x.view(-1, 784)
    lpx_z = px_z.log_prob(x).sum(-1)
    lpz = dist.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
    qz_x_ = qz_x.__class__(qz_x.loc.detach(), qz_x.scale.detach())
    lqz_x = qz_x_.log_prob(z).sum(-1)

    lw = lpz + lpx_z - lqz_x
    with torch.no_grad():
        reweight = torch.exp(lw - torch.logsumexp(lw, 0))
        z.register_hook(lambda grad: reweight.unsqueeze(-1) * grad)

    return (reweight * lw).sum(0).mean(0)


def train(epoch, train_loader, log_interval, model, lr=0.001, k=100):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in tqdm(enumerate(train_loader)):
        data = data.to(DEVICE)
        # Dynamic binarization
        sample = torch.rand(data.size()).to(DEVICE)
        data = (sample < data).float()

        optimizer.zero_grad()
        qz_x, px_z, z = model(data, k=k)
        # Use compute_elbo_dreg during training
        obj = compute_elbo_dreg
        loss = -obj(data, qz_x, px_z, z)
        loss.backward()
        train_loss += -loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    avg_loss = train_loss / (batch_idx + 1)  # type: ignore
    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, avg_loss))

    return avg_loss


def test(epoch, model, test_loader, k, batch_size):
    model.eval()
    test_loss = 0
    with torch.no_grad():

        for i, (data, _) in enumerate(test_loader):
            data = data.to(DEVICE)
            qz_x, px_z, z = model(data, k=k)
            test_loss += compute_elbo(data, qz_x, px_z, z).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], px_z.logits.view(batch_size * k, 1, 28, 28)[:n]]
                )
                save_image(
                    comparison.cpu(),
                    "./results/reconstruction_" + str(epoch) + ".png",
                    nrow=n,
                )

    test_loss /= i + 1  # type: ignore
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


def get_mnist(batch_size):

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data", train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return train_loader, test_loader


def main():

    batch_size = 64
    epochs = 10
    log_interval = 100
    model = VAE()
    model.to(DEVICE)

    train_loader, test_loader = get_mnist(batch_size=batch_size)

    train_stats, test_stats = [], []
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        train_stats.append(train(epoch, train_loader, log_interval, model))
        test_stats.append(
            test(epoch, model, test_loader, k=1000, batch_size=batch_size)
        )
        with torch.no_grad():
            sample = torch.randn(64, 20).to(DEVICE)
            sample = model.decode(sample).cpu()
            save_image(
                sample.view(64, 1, 28, 28), "./results/sample_" + str(epoch) + ".png"
            )

    return train_stats, test_stats


if __name__ == "__main__":

    main()
