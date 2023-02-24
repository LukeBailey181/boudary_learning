import numpy as np
from matplotlib import pyplot as plt 
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch

# ----- NEURAL NETWORK ----- #

class Net(nn.Module):

    def __init__(self, num_classes=2, hidden_size=20, input_dim=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def test_net(self, dataset, device):

        criterion = nn.CrossEntropyLoss()
        self.to(device)
        self.eval()
        total_loss = total_correct = total_examples = 0
        with torch.no_grad():
            for data in dataset:
                
                X,y = data 
                X = X.to(device)
                y = y.to(device)
                output = self(X)
                loss = criterion(output, y)
                total_loss += loss.item()
                total_correct += (output.argmax(dim=1) == y).sum().item()
                total_examples += len(y)

        return total_loss, total_correct / total_examples

    def train_net(self, epochs, trainset, device, lr=0.001):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr)
        losses = []

        self.train()
        self.to(device)
        for epoch in range(epochs):

            print(f"Epoch {epoch}")
            for data in trainset:
                X, y = data
                X = X.to(device)
                y = y.to(device)
                self.zero_grad()
                output = self(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

# ----- BOUNDARY GENERATION ----- #

def get_boundary(prob = 0.01, mean=[0,0], cov=[[1,0], [0,1]], num_points=1000):
    boundary_points = []
    normal_dist = multivariate_normal(mean=mean, cov=cov)

    # Fill boundary points
    while len(boundary_points) < num_points:
        draws = np.random.multivariate_normal(mean, cov, 10)
        draw_dens = normal_dist.pdf(draws)

        for draw, dens in zip(draws, draw_dens):
            if len(boundary_points) == num_points:
                break
            if dens <= prob:
                boundary_points.append(draw)

    x = [draw[0] for draw in boundary_points]
    y = [draw[1] for draw in boundary_points]

    plt.plot(x, y, '.')
    plt.axis('equal')
    plt.show()

    return boundary_points

def get_boundary_by_prob(prob = 0.01, mean=[0,0], cov=[[1,0], [0,1]], num_clust_points=10000, plot=False):
    boundary_points = []
    all_points = []
    normal_dist = multivariate_normal(mean=mean, cov=cov)

    # Fill boundary points
    draws = np.random.multivariate_normal(mean, cov, num_clust_points)
    draw_dens = normal_dist.pdf(draws)
    for draw, dens in zip(draws, draw_dens):
        all_points.append(draw)
        if dens <= prob:
            boundary_points.append(draw)

    if plot:
        x = [draw[0] for draw in boundary_points]
        y = [draw[1] for draw in boundary_points]

        plt.plot(x, y, '.')
        plt.axis('equal')

    return boundary_points, all_points

def get_dataloader_from_dataset(data, classes, batch_size=4):

    assert(len(data) == len(classes))
    dataset = []

    for boundary, class_ in zip(data, classes):
        for x in boundary:
            dataset.append([x.astype('float32'), class_])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    pass
