import random
from itertools import combinations

import networkx as nx

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

from utils.model import DrBC
from utils.dataflow import GraphData, TestData

BATCH_SIZE = 16
LOG_INTERVAL = 100
TOTAL_ITERATION = 10000

def wrap_data(data, dtype=None, cuda=True):
    data = torch.Tensor(data) if dtype is None else torch.tensor(data, dtype=dtype)
    data = data.cuda() if cuda else data
    return data

def calculate_loss(outs, label, source_ids, target_ids):
    pred = outs[source_ids] - outs[target_ids]
    gt = torch.sigmoid((label[source_ids] - label[target_ids]))

    loss = F.binary_cross_entropy_with_logits(pred, gt, reduction="sum")
    return loss

def calculate_metric(outs, label):
    topk = [1, 5, 10]
    k_accuracy = []
    node_nums = len(outs)

    label = label.reshape(-1)
    label = torch.argsort(label)
    outs = torch.argsort(outs)

    for k in topk:
        k_num = int(node_nums*k/100)
        k_label = label[:k_num].tolist()
        k_outs = outs[:k_num].tolist()

        correct = list(set(k_label) & set(k_outs))
        k_accuracy.append(len(correct)/(k_num))

    return k_accuracy


def train():
    model = DrBC()
    model = model.cuda()
    optimizer = Adam(params=model.parameters(), lr=0.0001)

    for iteration in range(TOTAL_ITERATION):
        if iteration % 500 == 0:
            print("Graph Generate")
            g = GraphData()

            edge_index = g.get_edge_index()
            train_data = g.get_train_data()
            label = g.get_label()

            edge_index = wrap_data(edge_index, dtype=torch.long)
            train_data = wrap_data(train_data)
            label = wrap_data(label)
            label = label.view(-1, 1)

        source_ids, target_ids = g.get_source_target_pairs()

        outs = model(train_data, edge_index)
        loss = calculate_loss(outs, label, source_ids, target_ids)

        if iteration % LOG_INTERVAL == 0:
            print("[{}/{}] Loss:{:.4f}".format(iteration, TOTAL_ITERATION, loss.item()))

        loss.backward()
        optimizer.step()

    test_graph = TestData("./hw1_data/Synthetic/5000/0_score.txt", "./hw1_data/Synthetic/5000/0.txt")
    val(model, test_graph)

def val(model, graph):
    edge_index = graph.get_edge_index()
    X = graph.get_train_data()
    label = graph.get_label()

    edge_index = wrap_data(edge_index, dtype=torch.long)
    X = wrap_data(X)
    label = wrap_data(label)
    
    outs = model(X, edge_index)

    print(calculate_metric(label, outs))




if __name__ == "__main__":
    train()

