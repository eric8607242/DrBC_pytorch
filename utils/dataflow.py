import random

import numpy as np
import networkx as nx

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch_geometric.utils import from_networkx


class GraphData:
    def __init__(self, batch_size=16):
        self.graph_list = [self._generate_graph() for i in range(batch_size)]

    def _generate_graph(self, num_max=200, num_min=100):
        node_nums = np.random.randint(num_max-num_min+1) + num_min
        g = nx.powerlaw_cluster_graph(n=node_nums, m=4, p=0.05)

        return g

    def _get_node_degree(self):
        degree_list = [graph.degree() for graph in self.graph_list]

        degree_value = []
        for degree in degree_list:
            degree_value.extend([v for _, v in degree])

        return degree_value

    def get_train_data(self):
        degree_value = self._get_node_degree()
        train_data = [[deg, 1, 1] for deg in degree_value]
        
        return train_data

    def get_edge_index(self):
        id_nums = 0
        edge_index = [[], []]
        for graph in self.graph_list:
            node_nums = len(graph.nodes)
            first_edge = [v1+id_nums for v1, v2 in graph.edges]
            second_edge = [v2+id_nums for v1, v2 in graph.edges]

            edge_index[0].extend(first_edge)
            edge_index[1].extend(second_edge)

            id_nums += node_nums

        return edge_index

    def get_label(self):
        bc_list = [list(nx.betweenness_centrality(graph, normalized=False).values()) for graph in self.graph_list]
        labels = []
        for bc in bc_list:
            labels.extend(bc)

        return labels

    def get_source_target_pairs(self, repeat=5):
        id_nums = 0
        source_ids = []
        target_ids = []
        
        for graph in self.graph_list:
            node_nums = len(graph.nodes)

            source_id = [i for i in range(id_nums, node_nums+id_nums)]
            target_id = [i for i in range(id_nums, node_nums+id_nums)]

            source_id *= 5
            target_id *= 5

            random.shuffle(source_id)
            random.shuffle(target_id)

            source_ids.extend(source_id)
            target_ids.extend(target_id)

            id_nums += node_nums
        return source_ids, target_ids


            
class TestData:
    def __init__(self, score_file, graph_file):
        self.label = self._read_file(score_file)
        self.edge_index = self._read_file(graph_file)

    def _read_file(self, file_path):
        fp = open(file_path, "r")

        data = []
        line = fp.readline()
        while line:
            line_txt = line[:-1].split("\t")
            data.append([float(t) for t in line_txt])
            line = fp.readline()

        fp.close()
        return data

    def _get_node_degree(self):
        degree_value = [0 for i in range(len(self.label))]

        for e in self.edge_index:
            degree_value[int(e[0])] += 1
        return degree_value
    
    def get_label(self):
        label = [bc[1] for bc in self.label]
        return label

    def get_edge_index(self):
        edge_index = [[], []]

        edge_index[0] = [int(e[0]) for e in self.edge_index]
        edge_index[1] = [int(e[1]) for e in self.edge_index]

        return edge_index

    def get_train_data(self):
        degree_value = self._get_node_degree()
        train_data = [[deg, 1, 1] for deg in degree_value]
        
        return train_data

if __name__ == "__main__":
    td = TestData("../hw1_data/Synthetic/5000/0_score.txt", "../hw1_data/Synthetic/5000/0.txt")
    print(td.get_edge_index())
