import numpy as np
import torch
from torch import nn


class LinearProjection(nn.Module):
    def __init__(self, in_dimension, out_dimension, weights_initialization="he", bias_initialization="zeros",
                 use_bias=True, scaling_factor=1, freeze=False):
        super().__init__()

        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.scaling_factor = scaling_factor
        self.use_bias = use_bias
        self.freeze = freeze

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
        self.weights.requires_grad = not freeze

        if use_bias:
            assert bias_initialization in ["zeros"]
            if bias_initialization == "zeros":
                bias = np.zeros(out_dimension, dtype=np.float32)

            self.bias = torch.from_numpy(bias)
            self.bias = nn.Parameter(self.bias)
            self.bias.requires_grad = not freeze

    def forward(self, x):
        # x.shape = [batch_size, in_dimension]
        # weights.shape = [in_dimension, out_dimension]
        _out = torch.matmul(x, self.weights * self.scaling_factor)
        if self.use_bias:
            _out = torch.add(_out, self.bias)
        return _out


class ReuseLinearProjection(LinearProjection):
    def __init__(self, layer, scaling_factor=1, transpose=False):
        if transpose and layer.use_bias:
            raise ValueError("Cannot set transpose flag while using bias from another layer")

        super().__init__(
            layer.in_dimension,
            layer.out_dimension,
            use_bias=layer.use_bias,
            freeze=layer.freeze,
            scaling_factor=scaling_factor
        )

        self.parent_layer = layer
        self.transpose = transpose

    def forward(self, x):  # todo: test if sharing weights works with backprop
        self.weights = self.parent_layer.weights.clone()
        if self.transpose:
            _out = torch.matmul(x, self.weights.T * self.scaling_factor)
        else:
            _out = torch.matmul(x, self.weights * self.scaling_factor)
        if self.use_bias:
            _out = torch.add(_out, self.bias)
        return _out


class PositionalEncodings:
    def __init__(self, out_dimension, device):
        self.out_dimension = out_dimension

        # generate positional encodings
        pos_vector = np.arange(0, out_dimension, 1)
        embeddings = np.zeros([out_dimension, out_dimension], dtype=np.float32)
        for i in range(out_dimension // 2):
            denom = 10000 ** (2 * i / out_dimension)
            embeddings[:, 2 * i] = np.sin(pos_vector / denom)
            embeddings[:, 2 * i + 1] = np.cos(pos_vector / denom)
        self.embeddings = torch.from_numpy(embeddings).to(device)

    def get_positional_encodings(self, sequence_lengths):
        """
        :param sequence_lengths: list of integers that represents len of sequences for the whole batch
        :return: list of torch.Tensor
        """
        out = []
        for sequence_length in sequence_lengths:
            out.append(
                self.embeddings[:sequence_length, :]
            )
        return out
