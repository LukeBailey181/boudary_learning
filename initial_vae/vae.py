from __future__ import print_function
import math
import torch
import torch.distributions as dist
from torch.nn import functional as F
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
import pickle
from collections import defaultdict

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

TRAIN_PATH = "../data/binary_MNIST/bin_mnist_train.pkl"
TEST_PATH = "../data/binary_MNIST/bin_mnist_test.pkl"


def load_binary_mnist(batch_size, train_path, test_path):
    with open(train_path, "rb") as f:
        trainset = pickle.load(f)
    with open(test_path, "rb") as f:
        testset = pickle.load(f)

    trainset = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainset, testset


class VAE(nn.Module):
    def __init__(
        self,
        dataset_name="mnist",
        latent_dim=20,
    ):
        """
        Args:
            dataset_name: one of {mnist, cifar10} indicating the dataset
                that the VAE will be trained on.
            latent_dim: dimension of latent space.
        """

        super(VAE, self).__init__()

        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num

        self.dataset_name = dataset_name
        if dataset_name == "mnist":
            self.fc1 = nn.Linear(784, 400)
            self.fc2 = nn.Linear(400, 400)
            self.fc31 = nn.Linear(400, latent_dim)
            self.fc32 = nn.Linear(400, latent_dim)
            self.fc4 = nn.Linear(latent_dim, 400)
            self.fc5 = nn.Linear(400, 400)
            self.fc6 = nn.Linear(400, 784)
        elif dataset_name == "cifar10":
            image_size = (32 * 32,)
            channel_num = (3,)
            kernel_num = (128,)

            # encoder
            self.encoder = nn.Sequential(
                self._conv(channel_num, kernel_num // 4),
                self._conv(kernel_num // 4, kernel_num // 2),
                self._conv(kernel_num // 2, kernel_num),
            )

            # encoded feature's size and volume
            self.feature_size = image_size // 8
            self.feature_volume = kernel_num * (self.feature_size**2)

            # q, end of encoder
            self.q_mean = self._linear(self.feature_volume, latent_dim, relu=False)
            self.q_logvar = self._linear(self.feature_volume, latent_dim, relu=False)

            # projection
            self.project = self._linear(latent_dim, self.feature_volume, relu=False)

            # decoder
            self.decoder = nn.Sequential(
                self._deconv(kernel_num, kernel_num // 2),
                self._deconv(kernel_num // 2, kernel_num // 4),
                self._deconv(kernel_num // 4, channel_num),
                nn.Sigmoid(),
            )

        else:
            raise ValueError(
                f"dataset_name of {dataset_name} not one of (mnist, cifar10)"
            )

    def encode(self, x):
        net = x

        if self.dataset_name == "mnist":
            net = F.relu(self.fc1(net))
            net = F.relu(self.fc2(net))
            return self.fc31(net), self.fc32(net).exp()
        elif self.dataset_name == "cifar10":
            encoded = self.encoder(net)
            unrolled = encoded.view(-1, self.feature_volume)
            mean, logvar = self.q_mean(unrolled), self.q_logvar(unrolled)
            return mean, logvar

    def decode(self, z):
        net = z

        if self.dataset_name == "mnist":
            net = F.relu(self.fc4(net))
            net = F.relu(self.fc5(net))
            return self.fc6(net)
        elif self.dataset_name == "cifar10":
            # TODO IMPLEMENT THIS
            pass

    def forward(self, x, k=1):
        if self.dataset_name == "mnist":
            x = x.view(-1, 784)
            qz_x = dist.Normal(*self.encode(x))
            z = qz_x.rsample(torch.Size([k]))
            px_z = dist.Bernoulli(logits=self.decode(z), validate_args=False)
            return qz_x, px_z, z
        elif self.dataset_name == "cifar10":
            # TODO CHANGE THIS
            x = x.view(-1, 784)
            qz_x = dist.Normal(*self.encode(x))
            z = qz_x.rsample(torch.Size([k]))

            """
            z_projected = self.project(z).view(
                -1, self.kernel_num,
                self.feature_size,
                self.feature_size,
            )
            """
            x_reconstructed = self.decoder(z)
            # What distribution to use for pz_x?

            pass

    # ---------------------- #
    # -----CIFAR LAYERS----- #
    # ---------------------- #

    def _conv(self, channel_size, kernel_num):
        """Conv layer for cifar10 encoder"""
        return nn.Sequential(
            nn.Conv2d(
                channel_size,
                kernel_num,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num):
        """Conv layer for cifar10 decoder"""
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num,
                kernel_num,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return (
            nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU(),
            )
            if relu
            else nn.Linear(in_size, out_size)
        )


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


def train(
    epoch,
    train_loader,
    log_interval,
    model,
    lr=0.001,
    k=100,
    dynamic_bin=False,
    verbose=False,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        # Dynamic binarization
        if dynamic_bin:
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

        if batch_idx % log_interval == 0 and verbose:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx,
                    len(train_loader),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    avg_loss = train_loss / (batch_idx + 1)  # type: ignore
    if verbose:
        print("====> Epoch: {} Average loss: {:.4f}".format(epoch, avg_loss))

    return avg_loss


def test(epoch, model, test_loader, k, batch_size, verbose=False):
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
    if verbose:
        print("====> Test set loss: {:.4f}".format(test_loss))

    return test_loss


def train_vae(
    train_loader,
    test_loader,
    batch_size,
    epochs,
    log_interval,
    save_path="./models/vae.pkl",
    k=1000,
    device=DEVICE,
):
    model = VAE()
    model.to(device)

    train_stats, test_stats = [], []
    for epoch in tqdm(range(1, epochs + 1)):
        # print(f"Epoch {epoch}")
        train_stats.append(train(epoch, train_loader, log_interval, model))
        test_stats.append(test(epoch, model, test_loader, k=k, batch_size=batch_size))
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(
                sample.view(64, 1, 28, 28), "./results/sample_" + str(epoch) + ".png"
            )

    torch.save(model, save_path)

    print("Train Loss:")
    print(train_stats)
    print("Test Loss:")
    print(test_stats)

    return model


def train_vae_on_mnist(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    batch_size=64,
    epochs=10,
    log_interval=100,
    save_path="./models/vae_mnist.pkl",
):
    train_loader, test_loader = load_binary_mnist(batch_size, train_path, test_path)

    train_vae(
        train_loader=train_loader,
        test_loader=test_loader,
        batch_size=batch_size,
        epochs=epochs,
        log_interval=log_interval,
        save_path=save_path,
    )


def train_class_spec_vaes(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    batch_size=64,
    epochs=30,
    log_interval=100,
    save_path="./models/vae_mnist.pkl",
):
    train_loader, test_loader = load_binary_mnist(batch_size, train_path, test_path)

    # Break train loader into class specific datasets
    train_sep = defaultdict(list)
    test_sep = defaultdict(list)

    for loader, class_sep in zip([train_loader, test_loader], [train_sep, test_sep]):
        for X, y in loader:
            for data_point, label in zip(X, y):
                class_sep[label.item()].append([data_point, label])

    classes = train_sep.keys()

    for class_ in classes:
        print(f"Training VAE on class {class_}")

        train_data = train_sep[class_]
        # Each class should have at least one piece of data
        assert class_ in test_sep
        test_data = test_sep[class_]

        class_train_loader = DataLoader(train_data, batch_size=batch_size)
        class_test_loader = DataLoader(test_data, batch_size=batch_size)

        train_vae(
            class_train_loader,
            class_test_loader,
            batch_size,
            epochs=epochs,
            log_interval=log_interval,
            save_path="./models/vae_" + str(class_) + ".pkl",
            k=1000,
        )


if __name__ == "__main__":
    # train_vae_on_mnist()
    train_class_spec_vaes(epochs=30)
