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
    sort_dataset_by_entropy,
    sort_dataset_by_p_p_comp,
    sort_dataset_by_top_prob_diff,
    random_prune,
    ordered_prune,
    flatten_dataset,
    load_class_spec_mnist,
    get_cifar10
)

from vae import compute_elbo, VAE, DEVICE, load_binary_mnist
from network import make_standard_net, train_net, test_net, make_dropoout_net, make_conv_net


TRAIN_PATH = "../data/binary_MNIST/bin_mnist_train.pkl"
TEST_PATH = "../data/binary_MNIST/bin_mnist_test.pkl"

ELBO_DATA_PATH = "../data/ordered_data/eblo.pkl"
CLASS_ELBO_DATA_PATH = "../data/ordered_data/class_eblo.pkl"
NEIGHBORS_DATA_PATH = "../data/ordered_data/latent_neighbors.pkl"
GAUSSIAN_DATA_PATH = "../data/ordered_data/gaussian_data.pkl"
ENTROPY_DATA_PATH = "../data/ordered_data/entropy_data.pkl"
TOP_PROB_DIFF_DATA_PATH = "../data/ordered_data/top_prob_diff_data,pkl"
P_P_COMP_DATA_PATH =  "../data/ordered_data/p_p_comp_data.plk"

MODEL_PATH = "./models/mnist_vae.pkl"
MODELS_ROOT_PATH = "./models/"

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


def train_and_test_on_dataset(
    train_dataset, test_loader, results, model_losses, run_key, lr, epochs, batch_size, num_classes=10, dataset_name="mnist"
):

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # ----- Train model with boundary -----#
    if dataset_name == "mnist":
        net = make_standard_net(
            num_classes=num_classes,
            input_dim=784,
            hidden_units=1200,
            hidden_layers=2,
        )
    elif dataset_name == "cifar10":
        net = make_conv_net()

    epoch_losses = train_net(epochs, net, train_loader, preproc=True, lr=lr, dataset_name=dataset_name)
    _, acc = test_net(net, test_loader)

    results[run_key].append(acc)
    model_losses[run_key].append(epoch_losses)

    return net


def test_boundary_learning(
    props, 
    repeats=3, 
    epochs=100, 
    batch_size=512, 
    k=1000, 
    lr=0.001, 
    print_results=False, 
    dataset_name="mnist",
    classes=None,
):

    vae_dict = load_mnist_vae_dict()
    multi_class_vae = torch.load(MODEL_PATH)
    # Collect elbo values

    if dataset_name == "mnist":
        if classes is None:
            trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
        else:
            trainset, testset = load_class_spec_mnist(TRAIN_PATH, TEST_PATH, classes=classes)
    elif dataset_name == "cifar10":
        if classes is None:
            trainset, testset = get_cifar10()
        else:
            raise NotImplementedError("Not yet implemented class specific cifar10 testing")


    if classes is None:
        num_classes=10
        #class_elbo_data, _ = sort_dataset_by_class_elbo(vae_dict, trainset, k, load_path=CLASS_ELBO_DATA_PATH)
        #elbo_data, _ = sort_dataset_by_elbo(multi_class_vae, trainset, k, load_path=ELBO_DATA_PATH)
        #nn_data, _ = sort_dataset_by_latent_neighbors(vae_dict, trainset, load_path=NEIGHBORS_DATA_PATH)
        #gaussian_data, _ = sort_dataset_by_fitted_gaussian(vae_dict, trainset, load_path=GAUSSIAN_DATA_PATH)
        entropy_data , _ = sort_dataset_by_entropy(trainset, testset, load_path=ENTROPY_DATA_PATH, dataset_name=dataset_name)
        p_p_comp_data, _ = sort_dataset_by_p_p_comp(trainset, testset, load_path=P_P_COMP_DATA_PATH, dataset_name=dataset_name)
        top_prob_diff_data, _ = sort_dataset_by_top_prob_diff(trainset, testset, load_path=TOP_PROB_DIFF_DATA_PATH, dataset_name=dataset_name)
    else:
        num_classes = len(classes)
        class_elbo_data, _ = sort_dataset_by_class_elbo(vae_dict, trainset, k)
        elbo_data, _ = sort_dataset_by_elbo(multi_class_vae, trainset, k)
        #nn_data, _ = sort_dataset_by_latent_neighbors(vae_dict, trainset)
        #gaussian_data, _ = sort_dataset_by_fitted_gaussian(vae_dict, trainset)
        entropy_data , _ = sort_dataset_by_entropy(trainset, testset, num_classes=num_classes)
        p_p_comp_data, _ = sort_dataset_by_p_p_comp(trainset, testset, num_classes=num_classes)
        top_prob_diff_data, _ = sort_dataset_by_top_prob_diff(trainset, testset, num_classes=num_classes)

    test_loader = torch.utils.data.DataLoader(
        flatten_dataset(testset),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    model_losses = defaultdict(list)
    results = defaultdict(list)
    for prop in props:

        train_data_dict = {}
        #train_data_dict["class_elbo_prune"] = flatten_dataset(ordered_prune(class_elbo_data, prop))
        #train_data_dict["class_elbo_prune_50_50"] = flatten_dataset(ordered_prune(class_elbo_data, prop, prop_random=0.5))
        #train_data_dict["elbo_prune"] = flatten_dataset(ordered_prune(elbo_data, prop))
        #train_data_dict["elbo_prune_50_50"] = flatten_dataset(ordered_prune(elbo_data, prop, prop_random=0.5))
        #train_data_dict["nn_prune"] = flatten_dataset(ordered_prune(nn_data, prop))
        #train_data_dict["gaussian_prune"] = flatten_dataset(ordered_prune(gaussian_data, prop))
        train_data_dict["entropy_prune"] = flatten_dataset(ordered_prune(entropy_data, prop))
        #train_data_dict["entropy_prune_50_50"] = flatten_dataset(ordered_prune(entropy_data, prop, prop_random=0.5))
        train_data_dict["p_p_comp_prune"] = flatten_dataset(ordered_prune(p_p_comp_data, prop))
        train_data_dict["top_prob_diff_prune"] = flatten_dataset(ordered_prune(top_prob_diff_data, prop))
        train_data_dict["random_prune"] = flatten_dataset(random_prune(elbo_data, prop))

        for rep in range(repeats):

            print(f"Testing proportion {prop}, repeat {rep+1}/{repeats}")
            for key, train_dataset in train_data_dict.items():

                # WTF IS THIS!
                train_and_test_on_dataset(
                    train_dataset,
                    test_loader,
                    results,
                    model_losses,
                    (key, prop),
                    lr,
                    epochs,
                    batch_size,
                    num_classes=num_classes,
                    dataset_name=dataset_name
                )

    if print_results:
        print(f"results = {dict(results)}")
        print(f"losses = {dict(model_losses)}")

    return results, model_losses


def visualize_pruned_data(k=1000, prop=0.01):

    vae_dict = load_mnist_vae_dict()
    multi_class_vae = torch.load(MODEL_PATH)

    # Collect sorted data
    trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    class_elbo_data, _ = sort_dataset_by_class_elbo(vae_dict, trainset, k=k, load_path=CLASS_ELBO_DATA_PATH)
    elbo_data, _ = sort_dataset_by_elbo(multi_class_vae, trainset, k=k, load_path=ELBO_DATA_PATH)
    nn_data, _ = sort_dataset_by_latent_neighbors(vae_dict, trainset, load_path=NEIGHBORS_DATA_PATH)
    gaussian_data, _ = sort_dataset_by_fitted_gaussian(vae_dict, trainset, load_path=GAUSSIAN_DATA_PATH)
    entropy_data , _ = sort_dataset_by_entropy(trainset, testset, load_path=ENTROPY_DATA_PATH)

    # Prune data
    pruned_data = {
        "class_elbo": ordered_prune(class_elbo_data, prop, shuffle=False),
        "elbo": ordered_prune(elbo_data, prop, shuffle=False),
        "nn": ordered_prune(nn_data, prop, shuffle=False),
        "gaussian": ordered_prune(gaussian_data, prop, shuffle=False),
        "random": random_prune(elbo_data, prop),
        "entropy": ordered_prune(entropy_data, prop, shuffle=False)
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

def test_metric_against_elbo(
        metric_func, epochs=150, batch_size=512, lr=0.0005, k=1000, use_testset=False, use_dropout=False, p=0.5
):
    """
    metric_func - logits are M dim, return should be scalar 
    """ 

    vae_dict = load_mnist_vae_dict()
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
    if use_dropout:
        net = make_dropoout_net(
            num_classes=10,
            input_dim=784,
            hidden_units=1200,
            hidden_layers=2,
            p=p
        )
    else:
        net = make_standard_net(
            num_classes=10,
            input_dim=784,
            hidden_units=1200,
            hidden_layers=2,
        )

    epoch_losses = train_net(epochs, net, train_loader, preproc=True, lr=lr)
    net.eval()
    _, acc = test_net(net, test_loader)
    print(f"MODEL ACCURACY: {acc}")

    if use_testset:
        class_elbo_data, class_elbos = sort_dataset_by_class_elbo(vae_dict, testset, k)
    else:
        class_elbo_data, class_elbos = sort_dataset_by_class_elbo(vae_dict, trainset, k, load_path=CLASS_ELBO_DATA_PATH)

    plotting_data = defaultdict(list)
    with torch.no_grad():
        net.to(DEVICE)
        dataset = flatten_dataset(trainset)

        print("Processing dataset")
        for elbo, (X, y) in tqdm(zip(class_elbos, dataset)):
            X = X.to(DEVICE)
            pred = net(X)
            logits = F.softmax(pred, dim=0).to("cpu")
            metric = metric_func(logits)

            if isinstance(y, torch.Tensor):
                y = y.item()
            plotting_data[y].append([elbo, metric])
        
    return plotting_data, epoch_losses

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
    print(f"MODEL ACCURACY: {acc}")

    class_elbo_data, class_elbos = sort_dataset_by_class_elbo(vae_dict, trainset, k, load_path=CLASS_ELBO_DATA_PATH)
    elbo_data, elbos = sort_dataset_by_elbo(multi_class_vae, trainset, k, load_path=ELBO_DATA_PATH)
    nn_data, distances = sort_dataset_by_latent_neighbors(vae_dict, trainset, load_path=NEIGHBORS_DATA_PATH)
    gaussian_data, likelihood = sort_dataset_by_fitted_gaussian(vae_dict, trainset, load_path=GAUSSIAN_DATA_PATH)

    # Get entropy for each point
    data = {
        "class_elbo": [class_elbo_data, class_elbos],
        "elbo": [elbo_data, elbos],
        "nn_data": [nn_data, distances],
        "gaussian_data": [gaussian_data, likelihood],
    }

    plotting_data = {}
    with torch.no_grad():
        net.to(DEVICE)
        for name, (dataset, metric) in data.items():

            dataset = flatten_dataset(dataset)

            entropy = []
            print(f"Process dataset {name}")
            for X, y in tqdm(dataset):
                X = X.to(DEVICE)
                pred = net(X)
                logits = F.softmax(pred, dim=0).to("cpu")
                entropy.append(-torch.sum(logits * torch.log(logits), dim=0).item())

            plotting_data[name] = [metric, entropy]

    return plotting_data, epoch_losses

def sort_datasets(k=1000):

    vae_dict = load_mnist_vae_dict()
    multi_class_vae = torch.load(MODEL_PATH)

    trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    sort_dataset_by_class_elbo(vae_dict, trainset, k, save_path=ELBO_DATA_PATH)
    sort_dataset_by_elbo(multi_class_vae, trainset, k, save_path=CLASS_ELBO_DATA_PATH)
    sort_dataset_by_latent_neighbors(vae_dict, trainset, save_path=NEIGHBORS_DATA_PATH)
    sort_dataset_by_fitted_gaussian(vae_dict, trainset, save_path=GAUSSIAN_DATA_PATH)
    sort_dataset_by_entropy(trainset, testset, save_path=ENTROPY_DATA_PATH)
    sort_dataset_by_p_p_comp(trainset, testset, save_path=P_P_COMP_DATA_PATH)
    sort_dataset_by_top_prob_diff(trainset, testset, save_path=TOP_PROB_DIFF_DATA_PATH)


def test_dataset_sort():

    trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    vae_dict = load_mnist_vae_dict()
    ordered_data, _ = sort_dataset_by_fitted_gaussian(vae_dict, trainset, k=1)
    pruned_data = ordered_prune(ordered_data, 0.5)


if __name__ == "__main__":

    #trainset, testset = load_binary_mnist(1, TRAIN_PATH, TEST_PATH)
    #sort_dataset_by_entropy(trainset, testset, save_path=ENTROPY_DATA_PATH)
    # test_dataset_sort()
    # visualize_pruned_data(k=1, prop=0.01)
    ##test_entropy_against_pruning_technique(epochs=150, k=1000)

    plotting_data, model_losses = test_boundary_learning(
        props=[1, 0.8, 0.6, 0.4, 0.2, 0.1],
        #props=[0.01],
        repeats=1,
        epochs=150,
        batch_size=512,
        k=1000,
        lr=0.001,
        dataset_name="cifar10"
        #mnist_classes=(0,1)
    )
    #sort_datasets()
    """
    test_boundary_learning(
        props=[0.98, 0.5, 0.1, 0.05, 0.01],
        repeats=2,
        epochs=150,
        batch_size=512,
        k=1000,
        lr=0.001,
    )
    """

    # visualize_latent_space()
