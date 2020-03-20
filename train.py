import random
from itertools import combinations

import networkx as nx

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

from utils.model import DrBC
def generate_graph():
    n = 20
    g = nx.powerlaw_cluster_graph(n=n, m=4, p=0.05)

    degree = g.degree([i for i in range(n)])
    degree = [i[1] for i in degree]

    bc = nx.betweenness_centrality(g)
    bc = [v for _, v in bc.items()]

    g = from_networkx(g)
    edge_index = g.edge_index

    return degree, bc, edge_index

def get_data(degree, bc):
    x = [[deg, 1, 1] for deg in degree]
    y = [[b] for b in bc]

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    return x, y

def get_batch_pair(gt, outs):
    node_nums = gt.shape[0]

    node_combine = list(combinations([i for i in range(node_nums)], 2))
    random.shuffle(node_combine)

    node_pairs = torch.zeros((len(node_combine), 2))
    for index, c in enumerate(node_combine):
        i, j = c
        node_pairs[index, 0] = gt[i]-gt[j]
        node_pairs[index, 1] = outs[i]-outs[j]

    return node_pairs



def train():
    model = DrBC()
    model = model.cuda()
    optimizer = Adam(params=model.parameters(), lr=0.001)

    #for e in range(1000):
    degree, bc, edge_index = generate_graph()
    x, y = get_data(degree, bc)
    x = x.cuda()
    y = y.cuda()
    edge_index = edge_index.cuda()
    for e in range(1000):

        batch_size = 16
        iteration_nums = len(x) // batch_size
        for i in range(iteration_nums):

            outs = model(x, edge_index)
            node_pairs = get_batch_pair(y, outs)

            data = node_pairs[i*batch_size:(i+1)*batch_size]
            print(node_pairs[0])
            loss = F.binary_cross_entropy(torch.sigmoid(data[:, 1]), torch.sigmoid(data[:, 0].detach()), reduction="sum")
            loss.backward()
            optimizer.step()




if __name__ == "__main__":
    train()

