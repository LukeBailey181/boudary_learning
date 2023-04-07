import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import numpy as np

from data import (
    sort_dataset_by_elbo,
    sort_dataset_by_class_elbo,
    sort_dataset_by_latent_neighbors,
    sort_dataset_by_fitted_gaussian,
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
    data_probs, _ = sort_dataset_by_elbo(vae, trainset, k)

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


def train_mlp_on_pruned_dataset(
    train_dataset, test_loader, results, model_losses, run_key, lr, epochs, batch_size
):

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
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
    epoch_losses = train_net(epochs, net, train_loader, preproc=True, lr=lr)
    _, acc = test_net(net, test_loader)

    results[run_key].append(acc)
    model_losses[run_key].append(epoch_losses)

    return net


def test_boundary_learning(
    props, repeats=3, epochs=100, batch_size=512, k=100, lr=0.001
):

    vae_dict = load_mnist_vae_dict()
    multi_class_vae = torch.load(MODEL_PATH)
    # Collect elbo values
    trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    class_elbo_data, _ = sort_dataset_by_class_elbo(vae_dict, trainset, k)
    elbo_data, _ = sort_dataset_by_elbo(multi_class_vae, trainset, k)
    nn_data, _ = sort_dataset_by_latent_neighbors(vae_dict, trainset)
    gaussian_data, _ = sort_dataset_by_fitted_gaussian(vae_dict, trainset)

    test_loader = torch.utils.data.DataLoader(
        flatten_dataset(testset),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    model_losses = defaultdict(list)
    results = defaultdict(list)
    for prop in props:

        class_elbo_pruned_train = flatten_dataset(ordered_prune(class_elbo_data, prop))
        elbo_pruned_train = flatten_dataset(ordered_prune(elbo_data, prop))
        nn_pruned_train = flatten_dataset(ordered_prune(nn_data, prop))
        gaussian_train = flatten_dataset(ordered_prune(gaussian_data, prop))
        random_train = flatten_dataset(random_prune(elbo_data, prop))

        for rep in range(repeats):

            print(f"Testing proportion {prop}, repeat {rep+1}/{repeats}")

            train_mlp_on_pruned_dataset(
                class_elbo_pruned_train,
                test_loader,
                results,
                model_losses,
                ("class_elbo_prune", prop),
                lr,
                epochs,
                batch_size,
            )
            train_mlp_on_pruned_dataset(
                elbo_pruned_train,
                test_loader,
                results,
                model_losses,
                ("elbo_prune", prop),
                lr,
                epochs,
                batch_size,
            )
            train_mlp_on_pruned_dataset(
                nn_pruned_train,
                test_loader,
                results,
                model_losses,
                ("latent_nn_prune", prop),
                lr,
                epochs,
                batch_size,
            )
            train_mlp_on_pruned_dataset(
                gaussian_train,
                test_loader,
                results,
                model_losses,
                ("gaussian_prune", prop),
                lr,
                epochs,
                batch_size,
            )
            train_mlp_on_pruned_dataset(
                random_train,
                test_loader,
                results,
                model_losses,
                ("random_prune", prop),
                lr,
                epochs,
                batch_size,
            )

    print(f"results = {dict(results)}")
    print(f"losses = {dict(model_losses)}")


def visualize_pruned_data(k=1000, prop=0.01):

    vae_dict = load_mnist_vae_dict()
    multi_class_vae = torch.load(MODEL_PATH)

    # Collect sorted data
    trainset, _ = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    class_elbo_data, _ = sort_dataset_by_class_elbo(vae_dict, trainset, k=k)
    elbo_data, _ = sort_dataset_by_elbo(multi_class_vae, trainset, k=k)
    nn_data, _ = sort_dataset_by_latent_neighbors(vae_dict, trainset)
    gaussian_data, _ = sort_dataset_by_fitted_gaussian(vae_dict, trainset)

    # Prune data
    pruned_data = {
        "class_elbo": ordered_prune(class_elbo_data, prop, shuffle=False),
        "elbo": ordered_prune(elbo_data, prop, shuffle=False),
        "nn": ordered_prune(nn_data, prop, shuffle=False),
        "gaussian": ordered_prune(gaussian_data, prop, shuffle=False),
        "random": random_prune(elbo_data, prop),
    }

    # Visualize`
    for name, data in pruned_data.items():
        save_image(
            [x[0].squeeze(0) for x in data],
            "./results/" + name + "_prop_" + str(prop) + ".png",
            normalize=True,
            nrow=20,
        )


def get_elbo_histogram_data(k=1000):

    vae_dict = load_mnist_vae_dict()
    multi_class_vae = torch.load(MODEL_PATH)

    trainset, _ = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    _, class_elbos = sort_dataset_by_class_elbo(vae_dict, trainset, k)
    _, elbos = sort_dataset_by_elbo(multi_class_vae, trainset, k)

    return {"class_elbos": class_elbos, "elbos": elbos}


def test_entropy_against_pruning_technique(
    epochs=150, batch_size=512, lr=0.0005, k=1000
):

    vae_dict = load_mnist_vae_dict()
    multi_class_vae = torch.load(MODEL_PATH)
    trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)

    train_loader = torch.utils.data.DataLoader(
        flatten_dataset(trainset),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        flatten_dataset(testset),
        batch_size=batch_size,
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

    class_elbo_data, class_elbos = sort_dataset_by_class_elbo(vae_dict, trainset, k)
    elbo_data, elbos = sort_dataset_by_elbo(multi_class_vae, trainset, k)
    nn_data, distances = sort_dataset_by_latent_neighbors(vae_dict, trainset)
    gaussian_data, likelihood = sort_dataset_by_fitted_gaussian(vae_dict, trainset)

    # Get entropy for each point
    data = {
        "class_elbo": [class_elbo_data, class_elbos],
        "elbo": [elbo_data, elbos],
        "nn_data": [nn_data, distances],
        "gaussian_data": [gaussian_data, likelihood],
    }

    plotting_data = {}
    with torch.no_grad():
        for name, (dataset, metric) in data.items():

            dataset = flatten_dataset(dataset)

            entropy = []
            print(f"Process dataset {name}")
            for X, y in tqdm(dataset):

                pred = net(X)
                logits = F.softmax(pred, dim=0)
                entropy.append(-torch.sum(logits * torch.log(logits), dim=0).item())

            plotting_data[name] = [metric, entropy]

    return plotting_data


def test_dataset_sort():

    trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    vae_dict = load_mnist_vae_dict()
    ordered_data, _ = sort_dataset_by_fitted_gaussian(vae_dict, trainset, k=1)
    pruned_data = ordered_prune(ordered_data, 0.5)


if __name__ == "__main__":

    # test_dataset_sort()
    # visualize_pruned_data(k=1, prop=0.01)
    test_entropy_against_pruning_technique(epochs=150, k=1000)

    """
    test_boundary_learning(
        props=[1, 0.5, 0.1, 0.05, 0.01],
        repeats=3,
        epochs=150,
        batch_size=512,
        k=1000,
        lr=0.001,
    )
    """

    """
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
