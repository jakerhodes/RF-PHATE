import pandas as pd
import numpy as np
# import torch

def dataprep(data, label_col_idx = 0, scale = 'normalize'):

    data = data.copy()
    categorical_cols = []
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'int64':
            categorical_cols.append(col)
            data[col] = pd.Categorical(data[col]).codes


    if label_col_idx is not None:
        label = data.columns[label_col_idx]
        y     = data.pop(label)
        x     = data


    if scale == 'standardize':
        for col in x.columns:
            # if col not in categorical_cols:
            if data[col].std() !=0:
                data[col] = (data[col] - data[col].mean()) / data[col].std()

    elif scale == 'normalize':
        for col in x.columns:
            # if col not in categorical_cols:
            if data[col].max() != data[col].min():
                data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    if label_col_idx is None:
        return np.array(x)
    else:
        return np.array(x), y

def tensor_dataset(data, device = 'cpu', **kwargs):
    x, y = dataprep(data, **kwargs)
    x = torch.tensor(np.array(x), dtype = torch.float)
    y = torch.tensor(np.array(y), dtype = torch.int64)

    return x.to(device), y.to(device)


def train_test_val_split(dataset, sizes = [.7, .15, .15], random_seed = None):

    """
    Returns the indices for training, test, and validation subsets
    """

    if random_seed is not None:
        torch.manual_seed(random_seed)

    n = len(dataset)
    train_size = np.int(np.ceil(n * sizes[0]))

    train_idx, test_val_idx = torch.utils.data.random_split(range(n), [train_size, n - train_size])

    test_size = np.int(np.ceil(((sizes[1] / (sizes[1] + sizes[2])) * len(test_val_idx))))
    val_size = len(test_val_idx) - test_size

    test_idx, val_idx = torch.utils.data.random_split(test_val_idx, [test_size, val_size])

    return list(train_idx), list(test_idx), list(val_idx)
