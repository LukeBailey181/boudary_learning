import torch
import torch.distributions as dist
from torch.nn import functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pickle


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


def load_binary_mnist(batch_size, train_path, test_path):

    with open(train_path, "rb") as f:
        trainset = pickle.load(f)
    with open(test_path, "rb") as f:
        testset = pickle.load(f)

    trainset = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainset, testset

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

    breakpoint()

    return [[i[0].to("cpu"), i[1]] for i in data_probs]

def random_prune(dataset, prop, num_classes=10):

    class_points = int((len(dataset) * prop) / num_classes)
    train_shuffle = random.sample(dataset, len(dataset))

    labels = [class_points] * num_classes

    pruned_data = []

    for X, y in train_shuffle:
        if labels[y.item()] > 0:
            pruned_data.append([X,y])
            labels[y.item()] -= 1

    return pruned_data

def elbo_prune(data_probs, prop, num_classes=10, reverse=False):

    data_dict = defaultdict(list)
    for X, y in data_probs:
        data_dict[y.item()].append([X,y])

    class_points = int((len(data_probs) * prop) / num_classes)

    pruned_data = []
    for class_ in range(num_classes):
        if reverse:
            pruned_data += data_dict[class_][-1 * class_points:]
        else:
            pruned_data += data_dict[class_][:class_points]

    return pruned_data

def flatten_dataset(dataset):
    output = []
    for X,y in dataset:
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

    trainset, testset = load_binary_mnist(64, train_path, test_path)
