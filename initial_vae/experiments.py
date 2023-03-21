import torch
from tqdm import tqdm
import random
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

from data import (
    sort_dataset_by_elbo,
    sort_dataset_by_class_elbo,
    sort_dataset_by_latent_neighbors,
    random_prune,
    ordered_prune,
    flatten_dataset,
)
from vae import compute_elbo, VAE, DEVICE, load_binary_mnist
from network import make_standard_net, train_net, test_net


TRAIN_PATH = "../data/binary_MNIST/bin_mnist_train.pkl"
TEST_PATH = "../data/binary_MNIST/bin_mnist_test.pkl"

MODEL_PATH = "./models/mnist_vae.pkl"
MODELS_ROOT_PATH = "./models/"


def test_vae_boundary_learning(
    props, repeats=3, epochs=100, batch_size=512, k=100, lr=0.001
):

    # Collect elbo values
    trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    vae = torch.load(MODEL_PATH)
    data_probs = sort_dataset_by_elbo(vae, trainset, k)

    test_loader = torch.utils.data.DataLoader(
        flatten_dataset(testset),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    model_losses = defaultdict(list)
    results = defaultdict(list)
    for prop in props:

        pruned_train = flatten_dataset(ordered_prune(data_probs, prop))
        random_train = flatten_dataset(random_prune(data_probs, prop))

        for rep in range(repeats):

            print(f"Testing proportion {prop}, repeat {rep+1}/{repeats}")
            boundary_loader = torch.utils.data.DataLoader(
                pruned_train,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )
            random_loader = torch.utils.data.DataLoader(
                random_train,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )

            # ----- Train model with boundary -----#
            net = make_standard_net(
                num_classes=10,
                input_dim=784,
                hidden_units=1200,
                hidden_layers=2,
            )
            epoch_losses = train_net(epochs, net, boundary_loader, preproc=True, lr=lr)

            _, boundary_acc = test_net(net, test_loader)

            results[("boundary", prop)].append(boundary_acc)
            model_losses[("boundary", prop)].append(epoch_losses)

            # ----- Train model with random sample -----#
            net = make_standard_net(
                num_classes=10,
                input_dim=784,
                hidden_units=1200,
                hidden_layers=2,
            )
            epoch_losses = train_net(epochs, net, random_loader, preproc=True, lr=lr)
            _, random_acc = test_net(net, test_loader)

            results[("random", prop)].append(random_acc)
            model_losses[("random", prop)].append(epoch_losses)

    print(f"results = {dict(results)}")
    print(f"losses = {dict(model_losses)}")


@torch.no_grad()
def get_latent_dataset(vae, dataset, k=1000):

    latent_datast = []
    for X, y in tqdm(dataset):
        X = X.to(DEVICE)
        qz_x, px_z, z = vae(X, k=k)
        latent_datast.append([z, y])

    Z = torch.cat([x[0] for x in latent_datast]).squeeze()
    y = [x[1].item() for x in latent_datast]

    return Z, y


def visualize_latent_space(k=1):

    trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    vae = torch.load(MODEL_PATH)
    Z, y = get_latent_dataset(vae, trainset, k=k)
    n_components = 2

    # -----Using PCA----- #
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(Z)

    total_var = pca.explained_variance_ratio_.sum() * 100
    print(f"Total explained variance: {total_var}")

    y = np.array(y)
    labels = np.array(list(range(10)))
    for label in labels:
        i = np.where(y == label)
        plt.scatter(components[i, 0], components[i, 1], label=label, s=1)

    plt.legend()
    plt.savefig("./results/latent_space_pca.png")
    plt.clf()

    # -----Using t-SNE----- #
    tsne = TSNE(n_components=2)
    compressed_z = tsne.fit_transform(Z)

    for label in labels:
        i = np.where(y == label)
        plt.scatter(compressed_z[i, 0], compressed_z[i, 1], label=label, s=1)

    plt.legend()
    plt.savefig("./results/latent_space_tsne.png")


# TODO make this robust to different path types
def load_mnist_vae_dict():

    vaes = {}
    vae = torch.load(MODEL_PATH)
    for class_ in range(10):
        model_path = MODELS_ROOT_PATH + "vae_" + str(class_) + ".pkl"
        vaes[class_] = torch.load(model_path)

    return vaes


def test_class_vaes_boundary_learning(
    props, repeats=3, epochs=100, batch_size=512, k=100, lr=0.001
):

    vae_dict = load_mnist_vae_dict()
    # Collect elbo values
    trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    data_probs = sort_dataset_by_class_elbo(vae_dict, trainset, k)

    test_loader = torch.utils.data.DataLoader(
        flatten_dataset(testset),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    model_losses = defaultdict(list)
    results = defaultdict(list)
    for prop in props:

        pruned_train = flatten_dataset(ordered_prune(data_probs, prop))
        random_train = flatten_dataset(random_prune(data_probs, prop))

        for rep in range(repeats):

            print(f"Testing proportion {prop}, repeat {rep+1}/{repeats}")
            boundary_loader = torch.utils.data.DataLoader(
                pruned_train,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )
            random_loader = torch.utils.data.DataLoader(
                random_train,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )

            # ----- Train model with boundary -----#
            net = make_standard_net(
                num_classes=10,
                input_dim=784,
                hidden_units=1200,
                hidden_layers=2,
            )
            epoch_losses = train_net(epochs, net, boundary_loader, preproc=True, lr=lr)

            _, boundary_acc = test_net(net, test_loader)

            results[("boundary", prop)].append(boundary_acc)
            model_losses[("boundary", prop)].append(epoch_losses)

            # ----- Train model with random sample -----#
            net = make_standard_net(
                num_classes=10,
                input_dim=784,
                hidden_units=1200,
                hidden_layers=2,
            )
            epoch_losses = train_net(epochs, net, random_loader, preproc=True, lr=lr)
            _, random_acc = test_net(net, test_loader)

            results[("random", prop)].append(random_acc)
            model_losses[("random", prop)].append(epoch_losses)

    print(f"results = {dict(results)}")
    print(f"losses = {dict(model_losses)}")

def test_nn_dataset_sort():

    trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    vae_dict = load_mnist_vae_dict()
    ordered_data = sort_dataset_by_latent_neighbors(vae_dict, trainset)
    pruned_data = ordered_prune(ordered_data, 0.5)

    breakpoint()

if __name__ == "__main__":

    test_nn_dataset_sort()

    """
    test_class_vaes_boundary_learning(
        props=[0.1],
        repeats=1,
        epochs=1,
        batch_size=512,
        k=1,
        lr=0.001,
    )
    test_vae_boundary_learning(
        props=[1, 0.5, 0.1, 0.05, 0.01],
        repeats=3,
        epochs=150,
        batch_size=512,
        k=1000,
        lr=0.001,
    )
    """

    # visualize_latent_space()
