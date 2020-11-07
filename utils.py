import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file

def prepare_data(dataset):
    filename =  dataset

    data = load_svmlight_file(filename)
    A, y = data[0], data[1]
    m, n = A.shape
    return A, y, m, n

def change_format(A, y, m):
    X = np.hstack((A.todense().astype(np.float32), np.ones((m, 1))))
    y = np.array((y - 1), dtype=int)
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.FloatTensor)
    return X, y


def weights_init_uniform(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    nn.init.zeros_(m.weight.data)


def test(model, loss_fn, X_test, y_test):
    with torch.no_grad():
        output = model(X_test)
        acc = accuracy_score(np.round(output.detach().numpy()), y_test.detach().numpy())
        loss = loss_fn(output, y_test.unsqueeze(dim=1)).item()
    return loss, acc


def train(model, loss_fn, optimizer, iter_num, X_train, y_train, X_test, y_test, batch_sz=10):  # , print_every=100
    init_loss, init_acc = test(model, loss_fn, X_test, y_test)
    losses = [init_loss]
    accuracies = [init_acc]

    for ep in range(iter_num + 1):

        nums = np.random.choice(range(X_train.shape[0]), batch_sz)
        X_batch = X_train[nums, :]
        y_batch = y_train[nums]

        model.train()
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch.unsqueeze(dim=1))
        loss.backward()
        optimizer.step()

        model.eval()
        loss, acc = test(model, loss_fn, X_test, y_test)
        accuracies.append(acc)
        losses.append(loss)
    return losses, accuracies
