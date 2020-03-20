import random
from itertools import combinations

import networkx as nx

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

from utils.model import DrBC
from utils.dataflow import GraphData

BATCH_SIZE = 16

def wrap_data(data, dtype=None, cuda=True):
    data = torch.tensor(data) if dtype is None else torch.tensor(data, dtype=dtype)
    data = data.cuda() if cuda else data
    return data

def calculate_loss(outs, label, source_ids, target_ids):
    pred = torch.sigmoid(outs[source_ids] - outs[target_ids])
    gt = torch.sigmoid(label[source_ids] - label[target_ids])

    loss = F.binary_cross_entropy(pred, gt, reduction="sum")
    return loss


def train():
    model = DrBC()
    model = model.cuda()
    optimizer = Adam(params=model.parameters(), lr=0.001)

    for iteration in range(5000):
        if iteration % 500 == 0:
            g = GraphData()

            edge_index = g.get_edge_index()

            train_data = self.get_train_data()
            label = g.get_label()

            edge_index = wrap_data(edge_index)
            train_data = wrap_data(train_data)
            label = wrap_data(label)

        source_ids, target_ids = g.get_source_target_pairs()

        outs = model(train_data, edge_index)
        loss = calculate_loss(outs, label, source_ids, target_ids)

        print(loss)
        loss.backward()
        optimizer.step()



if __name__ == "__main__":
    train()

