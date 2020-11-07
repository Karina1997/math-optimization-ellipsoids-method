import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class EllipsoidMethod(Optimizer):
    r"""Implements ellipsoid method."""

    def __init__(self, params, Q):
        defaults = dict(Q=Q)
        #         c = (Q[:, 0] + Q[:, 1]) / 2
        super().__init__(params, defaults)
        self.Q = Q
        self.n = Q.shape[0]
        R = (Q[:, 0] - Q[:, 1]) @ (Q[:, 0] - Q[:, 1]) / 2
        self.H = R ** 2 * torch.eye(self.n)

    def get_w(self, p):
        w = torch.zeros(1, self.n)
        w[p > self.Q[:, 1]] = 1
        w[p < self.Q[:, 0]] = -1

        return w

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if (p < self.Q[:, 0]).any() or (p > self.Q[:, 1]).any():
                    w_k = self.get_w(p)
                else:
                    w_k = p.grad

                _df = w_k / (np.sqrt(abs(w_k @ self.H @ w_k.t())))
                p.add_(_df @ self.H.t(), alpha=-1 / (self.n + 1))
                self.H = self.n ** 2 / (self.n ** 2 - 1) * (
                self.H - (2 / (self.n + 1)) * (self.H @ w_k.t() @ w_k @ self.H) / (w_k @ self.H @ w_k.t()))

        return loss
