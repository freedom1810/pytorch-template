import torch
import sklearn
import numpy as np
import copy 

def roc_auc_multi_label(output, target, label = 1):
    output = torch.softmax(output, dim = 1).numpy()
    target = target.numpy()

    labels = copy.deepcopy(target)

    if len(target.shape) == 1:
        target = target.astype(int)
        n_values = np.max(target) + 1
        target = np.eye(n_values)[target]
    
    # print(recall)
    train_predicts = output
    train_predicts_1 = train_predicts[:, 1].tolist()
    train_predicts_1_r = np.array(train_predicts_1) >= 0.5

    print('recall: ', sklearn.metrics.recall_score(labels, train_predicts_1_r))
    print('precision: ', sklearn.metrics.precision_score(labels, train_predicts_1_r))

    labels = labels.astype(int)
    train_predicts_1_r = train_predicts_1_r.astype(int)

    tp = sum(np.bitwise_and(labels, train_predicts_1_r))
    tn = len(train_predicts_1_r) - sum(np.bitwise_or(labels, train_predicts_1_r))
    print('tp {} tn {}'.format(tp, tn))

    return sklearn.metrics.roc_auc_score(target, output, multi_class = 'ovr', average=None)[1]

def roc_auc(output, target):
    output = torch.sigmoid(output.float())
    output = output.numpy()
    target = target.numpy()
    return sklearn.metrics.roc_auc_score(target, output)

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
