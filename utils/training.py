import numpy as np
import torch

from utils.utils import index_to_mask


def add_labels(feat, labels, idx):
    n_classes = (labels.max() + 1).item()
    onehot = torch.zeros([feat.shape[0], n_classes], device=feat.device)
    onehot[idx, labels[idx]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch, threshold_epoch=50):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * epoch / threshold_epoch


def random_planetoid_splits(data, num_classes, train_rate=0.6, val_rate=0.2, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    per_cls_train_lb = int(round(train_rate * len(data.y) / num_classes))
    val_lb = int(round(val_rate * len(data.y)))

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:per_cls_train_lb] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[per_cls_train_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=data.num_nodes)
        val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[per_cls_train_lb:per_cls_train_lb + val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[per_cls_train_lb + val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=data.num_nodes)
        val_mask = index_to_mask(val_index, size=data.num_nodes)
        test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return train_mask, val_mask, test_mask


def random_splits(data, train_rate=0.5, val_rate=0.25):
    labeled_nodes = torch.where(data.y != -1)[0]

    num_label = labeled_nodes.shape[0]
    train_num = int(num_label * train_rate)
    val_num = int(num_label * val_rate)

    perm = torch.randperm(num_label)

    train_indices = perm[:train_num]
    val_indices = perm[train_num: train_num + val_num]
    test_indices = perm[train_num + val_num:]

    train_mask = index_to_mask(labeled_nodes[train_indices], size=data.num_nodes)
    val_mask = index_to_mask(labeled_nodes[val_indices], size=data.num_nodes)
    test_mask = index_to_mask(labeled_nodes[test_indices], size=data.num_nodes)

    return train_mask, val_mask, test_mask


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label
