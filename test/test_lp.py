import numpy as np
import pytest
import torch

from transformer.blocks import LinearProjection

IN_DIMS = [16, 32, 128, 512]
OUT_DIMS = [16, 32, 128, 512]

WEIGHTS_INIT = ["glorot", "he", "ones"]
USE_BIAS = [True, False]

MEAN_EPS = 1e-1
STD_EPS = 1e-2


def _loop():
    for in_dims in IN_DIMS:
        for out_dims in OUT_DIMS:
            for weights_init in WEIGHTS_INIT:
                for use_bias in USE_BIAS:
                    yield in_dims, out_dims, weights_init, use_bias


def test_linear():
    for in_dims, out_dims, weights_init, use_bias in _loop():
        inp = np.random.uniform(0, 1, [in_dims, out_dims]).astype(np.float32)
        layer = LinearProjection(
            in_dims,
            out_dims,
            weights_initialization=weights_init,
            use_bias=use_bias
        )
        with torch.no_grad():
            out = layer(torch.from_numpy(inp)).numpy()
        if weights_init == "ones":
            stop = 1
        elif weights_init == "glorot":
            inp_mean = np.mean(inp)
            inp_std = np.std(inp)
            pred_mean = np.mean(out)
            pred_std = np.std(out)
            gt_mean = inp_mean + int(use_bias)
            fan_avg = np.mean([layer.in_dimension, layer.out_dimension])
            glorot_std = np.sqrt(1 / fan_avg)
            gt_std = inp_std * glorot_std
        elif weights_init == "he":
            stop = 1
        stop = 1
