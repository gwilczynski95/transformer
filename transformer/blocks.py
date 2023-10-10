import numpy as np
import torch
from torch import nn


class LinearProjection(nn.Module):
    def __init__(self, in_dimension, out_dimension, weights_initialization="he", bias_initialization="zeros",
                 use_bias=True):
        super().__init__()

        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self._use_bias = use_bias

        # initialize weights
        assert weights_initialization in ["he", "glorot", "normal", "ones"]
        if weights_initialization == "glorot":
            fan_avg = (self.in_dimension + self.out_dimension) / 2
            std = np.sqrt(1 / fan_avg)
        elif weights_initialization == "he":
            std = np.sqrt(2 / in_dimension)
        elif weights_initialization == "normal":
            std = 1
        weights = np.random.normal(0, std, [in_dimension, out_dimension]).astype(np.float32)
        if weights_initialization == "ones":
            weights = np.ones([in_dimension, out_dimension], dtype=np.float32)

        self.weights = torch.from_numpy(weights)
        self.weights = nn.Parameter(self.weights)

        if use_bias:
            assert bias_initialization in ["zeros"]
            if bias_initialization == "zeros":
                bias = np.zeros(out_dimension, dtype=np.float32)

            self.bias = torch.from_numpy(bias)
            self.bias = nn.Parameter(self.bias)

    def forward(self, x):
        # x.shape = [batch_size, in_dimension]
        # weights.shape = [in_dimension, out_dimension]
        _out = torch.matmul(x, self.weights)
        if self._use_bias:
            _out = torch.add(_out, self.bias)
        return _out
