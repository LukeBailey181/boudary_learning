import torch
import torch.distributions as dist
from torch.nn import functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pickle
import random
from tqdm import tqdm
from vae import compute_elbo, DEVICE
from collections import defaultdict
import numpy as np
from scipy.stats import multivariate_normal


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


def _binarize_dataloader(dataloader):

    binary_dataset = []
    for X, y in dataloader:
        sample = torch.rand(X.size())
        bin_X = (sample < X).float()
        batch = [bin_X.squeeze(0), y.item()]
        binary_dataset.append(batch)

    return binary_dataset


# TODO make this directory saving more robust using path objects
def gen_binary_mnist(train_path, test_path):

    train_loader, test_loader = get_mnist(1)

    trainset = _binarize_dataloader(train_loader)
    testset = _binarize_dataloader(test_loader)

    with open(train_path, "wb") as f:
        pickle.dump(trainset, f)
    with open(test_path, "wb") as f:
        pickle.dump(testset, f)


@torch.no_grad()
def sort_dataset_by_elbo(vae, dataset, k=100):

    vae.to(DEVICE)

    data_probs = []
    for X, y in tqdm(dataset):
        # Assert batch size is 1
        assert X.shape[0] == 1

        X = X.to(DEVICE)
        qz_x, px_z, z = vae(X, k=k)
        elbo = compute_elbo(X, qz_x, px_z, z)
        data_probs.append([X, y, elbo])

    data_probs.sort(key=lambda x: x[2])

    return [[i[0].to("cpu"), i[1]] for i in data_probs]


@torch.no_grad()
def sort_dataset_by_class_elbo(vae_dict, dataset, k=100):

    for vae in vae_dict.values():
        vae.to(DEVICE)

    data_probs = []
    for X, y in tqdm(dataset):
        # Assert batch size is 1
        assert X.shape[0] == 1

        X = X.to(DEVICE)
        vae = vae_dict[y.item()]

        qz_x, px_z, z = vae(X, k=k)
        elbo = compute_elbo(X, qz_x, px_z, z)
        data_probs.append([X, y, elbo])

    data_probs.sort(key=lambda x: x[2])

    return [[i[0].to("cpu"), i[1]] for i in data_probs]


@torch.no_grad()
def sort_dataset_by_fitted_gaussian(vae_dict, dataset):

    for vae in vae_dict.values():
        vae.to(DEVICE)

    embeddings = defaultdict(list)
    for X, y in tqdm(dataset):
        # Assert batch size is 1
        assert X.shape[0] == 1

        X = X.to(DEVICE)
        vae = vae_dict[y.item()]
        qz_x, px_z, z = vae(X, k=1)
        embeddings[y.item()].append([X, y, z.squeeze().to("cpu")])

    # Calculate centroids and covariances
    gaussians = {}
    for (
        class_,
        class_data,
    ) in embeddings.items():
        # concatenate and find mean along the first dimension
        embds = [i[2] for i in class_data]
        embds = torch.stack(embds)
        mean = torch.mean(embds, dim=0)
        cov = np.cov(np.array(embds).T)
        gaussians[class_] = multivariate_normal(mean=mean, cov=cov)

    data_probs = []
    for X, y in tqdm(dataset):
        # Assert batch size is 1

        X = X.to(DEVICE)

        gaussian = gaussians[y.item()]
        vae = vae_dict[y.item()]
        _, _, z = vae(X, k=1)
        dens = gaussian.pdf(z.squeeze().to("cpu"))

        data_probs.append([X, y, dens])
        
    # Sort with largest distances first
    data_probs.sort(key=lambda x: x[2])

    return [[i[0].to("cpu"), i[1]] for i in data_probs]


@torch.no_grad()
def sort_dataset_by_latent_neighbors(vae_dict, dataset):
    """Returns dataset sorted by distance of points to centroid
    of latent space cluster corresponding to that class"""

    for vae in vae_dict.values():
        vae.to(DEVICE)

    embeddings = defaultdict(list)
    for X, y in tqdm(dataset):
        # Assert batch size is 1
        assert X.shape[0] == 1

        X = X.to(DEVICE)
        vae = vae_dict[y.item()]
        qz_x, px_z, z = vae(X, k=1)
        embeddings[y.item()].append([X, y, z.squeeze()])

    # Calculate centroids
    centroids = {}
    for (
        class_,
        class_data,
    ) in embeddings.items():
        # concatenate and find mean along the first dimension
        embds = [i[2] for i in class_data]
        centroids[class_] = torch.mean(torch.stack(embds), dim=0)

    # Find distances to cetroids
    data_distances = []
    for class_, class_data in embeddings.items():
        for X, y, z in class_data:
            distance = (z - centroids[class_]).pow(2).sum().sqrt()
            assert distance.item() >= 0
            data_distances.append([X, y, distance.item()])

    # Sort with largest distances first
    data_distances.sort(key=lambda x: x[2], reverse=True)

    return [[i[0].to("cpu"), i[1]] for i in data_distances]


def random_prune(dataset, prop, num_classes=10):

    class_points = int((len(dataset) * prop) / num_classes)
    train_shuffle = random.sample(dataset, len(dataset))

    labels = [class_points] * num_classes

    pruned_data = []

    for X, y in train_shuffle:
        if labels[y.item()] > 0:
            pruned_data.append([X, y])
            labels[y.item()] -= 1

    return pruned_data


def ordered_prune(sorted_data, prop, num_classes=10, reverse=False, shuffle=True):

    data_dict = defaultdict(list)
    for idx, (X, y) in enumerate(sorted_data):
        data_dict[y.item()].append([X, y, idx])

    class_points = int((len(sorted_data) * prop) / num_classes)

    pruned_data = []
    for class_ in range(num_classes):
        if reverse:
            pruned_data += data_dict[class_][-1 * class_points :]
        else:
            pruned_data += data_dict[class_][:class_points]

    # Sort by index to preserver original ordering
    pruned_data.sort(key=lambda x: x[2])

    # Remove indexes
    pruned_data = [[i[0], i[1]] for i in pruned_data]

    # Shuffle data
    if shuffle:
        pruned_data = random.sample(pruned_data, len(pruned_data))

    return pruned_data


def flatten_dataset(dataset):
    output = []
    for X, y in dataset:
        output.append([X.flatten(), y.item()])
    return output


if __name__ == "__main__":

    data_dir = "../data/binary_MNIST/"
    train_path = data_dir + "bin_mnist_train.pkl"
    test_path = data_dir + "bin_mnist_test.pkl"

    gen_binary_mnist(
        train_path=train_path,
        test_path=test_path,
    )
