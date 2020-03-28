def calculate_metric(outs, label):
    topk = [1, 5, 10]
    k_accuracy = []
    node_nums = len(outs)

    label = label.reshape(-1)
    tau, _ = stats.kendalltau(outs, label)

    print("Kendall tau: {}".format(tau))

    label = torch.argsort(label)
    outs = torch.argsort(outs)

    for k in topk:
        k_num = int(node_nums*k/100)
        k_label = label[:k_num].tolist()
        k_outs = outs[:k_num].tolist()

        correct = list(set(k_label) & set(k_outs))
        k_accuracy.append(len(correct)/(k_num))
        print("Top-{} accuracy: {}".format(k, k_accuracy*100))

    return k_accuracy

def calculate_loss(outs, label, source_ids, target_ids):
    pred = outs[source_ids] - outs[target_ids]
    gt = torch.sigmoid((label[source_ids] - label[target_ids]))

    loss = F.binary_cross_entropy_with_logits(pred, gt, reduction="sum")
    return loss

def wrap_data(data, dtype=None, cuda=True):
    data = torch.Tensor(data) if dtype is None else torch.tensor(data, dtype=dtype)
    data = data.cuda() if cuda else data
    return data

def load(model, load_path):
    state_dict = torch.load(load_path)
    model_dict = model.state_dict()

    state_dict = {K:v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)

    model.load_state_dict(model_dict)
