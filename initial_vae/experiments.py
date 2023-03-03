import torch
from tqdm import tqdm
import random
from collections import defaultdict

from data import load_binary_mnist
from vae import compute_elbo, VAE, DEVICE
from network import make_standard_net, train_net, test_net


TRAIN_PATH = "../data/binary_MNIST/bin_mnist_train.pkl"
TEST_PATH = "../data/binary_MNIST/bin_mnist_test.pkl"

MODEL_PATH = "./models/mnist_vae.pkl"

@torch.no_grad()
def sort_dataset_by_elbo(vae, dataset, k=100):

    vae.to(DEVICE)

    data_probs = []
    for X, y in tqdm(dataset):
        X = X.to(DEVICE)
        qz_x, px_z, z = vae(X, k=k)
        elbo = compute_elbo(X, qz_x, px_z, z)
        data_probs.append([X, y, elbo])

    data_probs.sort(key=lambda x: x[2])

    return [[i[0].to("cpu"), i[1]] for i in data_probs]

def elbo_prune(data_probs, prop):

    num_points = int(len(data_probs) * prop)
    return (data_probs[:num_points])

def flatten_dataset(dataset):
    output = []
    for X,y in dataset:
        output.append([X.flatten(), y.item()])
    return output

def test_vae_boundary_learning(
    props,
    repeats=3,
    epochs=100,
    batch_size=512,
    k=100,
    lr=0.001
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

        pruned_train = flatten_dataset(elbo_prune(data_probs, prop))
        random_train = random.sample(flatten_dataset(trainset), int(len(trainset) * prop))
       
        for _ in range(repeats):
            
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

            #----- Train model with boundary -----#
            net = make_standard_net(
                num_classes=10,
                input_dim=784,
                hidden_units=1200,
                hidden_layers=2,
            )
            epoch_losses = train_net(
                epochs, net, boundary_loader, preproc=True, lr=lr
            )

            _, standard_acc = test_net(net, test_loader)

            results[("boundary", prop)].append(standard_acc)
            model_losses[("boundary", prop)].append(epoch_losses)

            #----- Train model with random sample -----#
            net = make_standard_net(
                num_classes=10,
                input_dim=784,
                hidden_units=1200,
                hidden_layers=2,
            )
            epoch_losses = train_net(
                epochs, net, random_loader, preproc=True, lr=lr
            )
            _, dropout_acc = test_net(net, test_loader)

            results[("random", prop)].append(dropout_acc)
            model_losses[("random", prop)].append(epoch_losses)

    print(f"results = {dict(results)}")
    print(f"losses = {dict(model_losses)}")

if __name__ == "__main__":


    test_vae_boundary_learning(
        props = [1, 0.5, 0.1, 0.05, 0.01],
        repeats=3,
        epochs=150,
        batch_size=512,
        k=1000,
        lr=0.001
    )
