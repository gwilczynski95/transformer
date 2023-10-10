import numpy as np
import torch
from torch import nn


class LinearLayer(nn.Module):
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


class ReuseLinearLayer(LinearLayer):
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


class PositionWiseFeedForward(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, weights_initialization="he",
                 bias_initialization="zeros"):
        super().__init__()
        self.layer1 = LinearLayer(
            in_dim,
            mid_dim,
            weights_initialization=weights_initialization,
            bias_initialization=bias_initialization,
            use_bias=True
        )
        self.layer2 = LinearLayer(
            mid_dim,
            out_dim,
            weights_initialization=weights_initialization,
            bias_initialization=bias_initialization,
            use_bias=True
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        _out = self.layer1(x)
        _out = self.relu(_out)
        return self.layer2(_out)


class ScaledDotProductAttention(nn.Module):  # todo: test this
    def __init__(self, inp_dim, keys_dim, values_dim, weights_initialization="he"):
        super().__init__()

        self.keys_dim = keys_dim

        self.layer_Q = LinearLayer(
            inp_dim,
            keys_dim,
            weights_initialization=weights_initialization,
            use_bias=False
        )
        self.layer_K = LinearLayer(
            inp_dim,
            keys_dim,
            weights_initialization=weights_initialization,
            use_bias=False
        )
        self.layer_V = LinearLayer(
            inp_dim,
            values_dim,
            weights_initialization=weights_initialization,
            use_bias=False
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        # inp [batch_size, num_of_tokens, inp_dim]
        out = []  # todo: don't know if this is ok
        for batch_idx in x.shape[0]:
            Q = self.layer_Q(x[batch_idx, :, :])
            K = self.layer_K(x[batch_idx, :, :])
            V = self.layer_V(x[batch_idx, :, :])
            soft_out = self.softmax(
                torch.matmul(Q, K.T) / torch.sqrt(self.keys_dim)
            )
            tokens = torch.matmul(soft_out, V)
            out.append(tokens)
        return out


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
