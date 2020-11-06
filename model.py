import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, input_dim, result_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, result_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        return self.sigmoid(self.lin(X))