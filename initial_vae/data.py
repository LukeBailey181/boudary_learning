import torch
import torch.distributions as dist
from torch.nn import functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions import Categorical
from torchvision.utils import save_image
import pickle
import random
from tqdm import tqdm
from vae import compute_elbo, DEVICE
from collections import defaultdict
import numpy as np
from scipy.stats import multivariate_normal
from network import make_standard_net, train_net, test_net


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
def sort_dataset_by_elbo(vae, dataset, k=100, load_path=None, save_path=None):

    if load_path is not None:
        with open(load_path, "rb") as f:
            return pickle.load(f)

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

    output = [[i[0].to("cpu"), i[1]] for i in data_probs], [i[-1] for i in data_probs]
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(output, f)

    return output


@torch.no_grad()
def sort_dataset_by_class_elbo(vae_dict, dataset, k=100, load_path=None, save_path=None):

    if load_path is not None:
        with open(load_path, "rb") as f:
            return pickle.load(f)

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

    output = [[i[0].to("cpu"), i[1]] for i in data_probs], [i[-1] for i in data_probs]
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(output, f)

    return output

@torch.no_grad()
def sort_dataset_by_fitted_gaussian(vae_dict, dataset, load_path=None, save_path=None):

    if load_path is not None:
        with open(load_path, "rb") as f:
            return pickle.load(f)

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

    output = [[i[0].to("cpu"), i[1]] for i in data_probs], [i[-1] for i in data_probs]
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(output, f)

    return output



@torch.no_grad()
def sort_dataset_by_latent_neighbors(vae_dict, dataset, load_path=None, save_path=None):
    """Returns dataset sorted by distance of points to centroid
    of latent space cluster corresponding to that class"""

    if load_path is not None:
        with open(load_path, "rb") as f:
            return pickle.load(f)

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

    output = [[i[0].to("cpu"), i[1]] for i in data_distances], [i[-1] for i in data_distances]
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(output, f)

    return output

def sort_dataset_by_entropy(trainset, testset, load_path=None, save_path=None, epochs=150, lr=0.005):
    return sort_dataset_by_model_metric(trainset, testset, calc_entropy, True, load_path, save_path, epochs, lr)

def sort_dataset_by_p_p_comp(trainset, testset, load_path=None, save_path=None, epochs=150, lr=0.005):
    return sort_dataset_by_model_metric(trainset, testset, calc_p_p_complement_sum, True, load_path, save_path, epochs, lr)

def sort_dataset_by_top_prob_diff(trainset, testset, load_path=None, save_path=None, epochs=150, lr=0.005):
    return sort_dataset_by_model_metric(trainset, testset, calc_top_prob_diff, False, load_path, save_path, epochs, lr)

def sort_dataset_by_model_metric(trainset, testset, metric_func, reverse_sort, load_path=None, save_path=None, epochs=150, lr=0.005):

    if load_path is not None:
        with open(load_path, "rb") as f:
            return pickle.load(f)

    # Train model on dataset
    train_loader = torch.utils.data.DataLoader(
        flatten_dataset(trainset),
        batch_size=512,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        flatten_dataset(testset),
        batch_size=512,
        shuffle=True,
        pin_memory=True,
    )

    # ----- Train model -----#
    net = make_standard_net(
        num_classes=10,
        input_dim=784,
        hidden_units=1200,
        hidden_layers=2,
    )
    net.eval()
    epoch_losses = train_net(epochs, net, train_loader, preproc=True, lr=lr)
    _, acc = test_net(net, test_loader)
    print(f"MODEL ACCURACY: {acc}")

    data_metric = []
    with torch.no_grad():
        for X, y in tqdm(trainset):
            X_flat = X.flatten()
            X_flat = X_flat.to(DEVICE)
            pred = net(X_flat)
            logits = F.softmax(pred, dim=0).to("cpu")
            metric = metric_func(logits)
            data_metric.append([X,y, metric])

    data_metric.sort(key=lambda x: x[2], reverse=reverse_sort)

    output = [[i[0].to("cpu"), i[1]] for i in data_metric], [i[-1] for i in data_metric]
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(output, f)

    return output

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


def ordered_prune(sorted_data, prop, num_classes=10, reverse=False, shuffle=True, prop_random=0):

    data_dict = defaultdict(list)
    for idx, (X, y) in enumerate(sorted_data):
        data_dict[y.item()].append([X, y, idx])

    class_points = int((len(sorted_data) * prop  * (1 - prop_random)) / num_classes)
    random_points = int((len(sorted_data) * prop * prop_random) / num_classes)

    pruned_data = []
    for class_ in range(num_classes):
        if reverse:
            pruned_data += data_dict[class_][-1 * class_points :]
            #pruned_data += random.sample(data_dict[class_][: -1 * class_points], random_points)
            pruned_data += random.sample(data_dict[class_], random_points)
        else:
            pruned_data += data_dict[class_][:class_points]
            #pruned_data += random.sample(data_dict[class_][class_points:], random_points)
            pruned_data += random.sample(data_dict[class_], random_points)

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


def calc_entropy(logits):
    """
    Keyword aguments:
        logits - M dim tensor corresponding to logits for single input 
    Returns:
        Scalar entropy value
    """
    return Categorical(probs = logits).entropy().item()

def calc_p_p_complement_sum(logits):
    """
    Keyword aguments:
        logits - M dim tensor corresponding to logits for single input 
    Returns:
        Scalar value corresponding to sum_c p_c (1-p_c)
    """

    return torch.sum(logits * (1 - logits)).item()

def calc_top_prob_diff(logits):
    """
    Keyword aguments:
        logits - M dim tensor corresponding to logits for single input 
    Returns:
        Scalar value corresponding to difference between two highest values 
        in logits
    """
    sorted_p, _ = torch.sort(logits, descending=True)
    p_1, p_2 = sorted_p[0], sorted_p[1]

    if (p_1 - p_2).item() < 0:
        print(p_1)
        print(p_2)
        print(logits)
        print(sorted_p)
        raise(ValueError)

    return (p_1 - p_2).item()

if __name__ == "__main__":

    data_dir = "../data/binary_MNIST/"
    train_path = data_dir + "bin_mnist_train.pkl"
    test_path = data_dir + "bin_mnist_test.pkl"

    gen_binary_mnist(
        train_path=train_path,
        test_path=test_path,
    )
