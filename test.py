import argparse

import torch

from utils.model import DrBC
from utils.dataflow import GraphData, TestData
from utils.utils import load
from train import val

MODEL_PATH = "./model.pth"


def test(score_path, data_path):
    model = DrBC()
    model.cuda()

    load(model, MODEL_PATH)
    model = model.cpu()

    test_graph = TestData(score_path, data_path)
    val(model, test_graph, cuda=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str, help="path to bc score", required=True)
    parser.add_argument("--data_path", type=str, help="path to bc data", required=True)
    args = parser.parse_args()

    test(args.score_path, args.data_path)

