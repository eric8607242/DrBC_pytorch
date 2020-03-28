import random
import argparse
from itertools import combinations

import networkx as nx

import torch
from torch.optim import Adam
from torch_geometric.utils import from_networkx

import scipy.stats as stats

from utils.model import DrBC
from utils.dataflow import GraphData, TestData
from utils.utils import calculate_metric, wrap_data, calculate_loss

LOG_INTERVAL = 100
TOTAL_ITERATION = 10000
MODEL_PATH = "./model.pth"

def train(score_path, data_path):
    model = DrBC()
    model = model.cuda()
    optimizer = Adam(params=model.parameters(), lr=0.0001)

    val_graph = GraphData(batch_size=1)

    for iteration in range(TOTAL_ITERATION):
        if iteration % 500 == 0:
            print("-----------------Val---------------------")
            val(model, val_graph)

            print("Graph Generate")
            graph = GraphData()

            print("-----------------Train---------------------")
            edge_index = graph.get_edge_index()
            train_data = graph.get_train_data()
            label = graph.get_label()

            edge_index = wrap_data(edge_index, dtype=torch.long)
            train_data = wrap_data(train_data)
            label = wrap_data(label)
            label = label.view(-1, 1)
        model.train()

        source_ids, target_ids = graph.get_source_target_pairs()

        outs = model(train_data, edge_index)
        loss = calculate_loss(outs, label, source_ids, target_ids)

        if iteration % LOG_INTERVAL == 0:
            print("[{}/{}] Loss:{:.4f}".format(iteration, TOTAL_ITERATION, loss.item()))

        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)

    print("-----------------Test---------------------")
    test_graph = TestData(score_path, data_path)
    val(model, test_graph)

def val(model, graph, cuda=True):
    model.eval()
    edge_index = graph.get_edge_index()
    X = graph.get_train_data()
    label = graph.get_label()

    
    edge_index = wrap_data(edge_index, dtype=torch.long, cuda=cuda)
    X = wrap_data(X, cuda=cuda)
    label = wrap_data(label, cuda=False)
    
    with torch.no_grad():
        outs = model(X, edge_index)

    calculate_metric(outs, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str, help="path to bc score", required=True)
    parser.add_argument("--data_path", type=str, help="path to bc data", required=True)
    args = parser.parse_args()

    train(args.score_path, args.data_path)

