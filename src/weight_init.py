import math
import numpy as np
from mindspore import Tensor, Parameter

def gru_default_state(batch_size, input_size, hidden_size, num_layers=1, bidirectional=True):
    '''Weight init for gru cell'''
    stdv = 1 / math.sqrt(hidden_size)
    weight_i = Parameter(Tensor(
        np.random.normal(0.0, 0.02, (input_size, 3*hidden_size)).astype(np.float16)), name='weight_i')
    weight_h = Parameter(Tensor(
        np.random.normal(0.0, 0.02, (hidden_size, 3*hidden_size)).astype(np.float16)), name='weight_h')
    bias_i = Parameter(Tensor(
        np.random.normal(0.0, 0.02, (3*hidden_size)).astype(np.float16)), name='bias_i')
    bias_h = Parameter(Tensor(
        np.random.normal(0.0, 0.02, (3*hidden_size)).astype(np.float16)), name='bias_h')
    init_h = Tensor(np.zeros((batch_size, hidden_size)).astype(np.float16))
    return weight_i, weight_h, bias_i, bias_h, init_h

def dense_default_state(in_channel, out_channel):
    '''Weight init for dense cell'''
    stdv = 1 / math.sqrt(in_channel)
    weight = Tensor(np.random.uniform(-stdv, stdv, (out_channel, in_channel)).astype(np.float16))
    bias = Tensor(np.random.uniform(-stdv, stdv, (out_channel)).astype(np.float16))
    return weight, bias

def _average_units(shape):
    """
    Average shape dim.
    """
    if not shape:
        return 1.
    if len(shape) == 1:
        return float(shape[0])
    if len(shape) == 2:
        return float(shape[0] + shape[1]) / 2.
    raise RuntimeError("not support shape.")

def weight_variable(shape):
    scale_shape = shape
    avg_units = _average_units(scale_shape)
    scale = 1.0 / max(1., avg_units)
    limit = math.sqrt(3.0 * scale)
    values = np.random.uniform(-limit, limit, shape).astype(np.float32)
    return Tensor(values)

def one_weight(shape):
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)

def zero_weight(shape):
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)

def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units**-0.5, shape).astype(np.float32)
    return Tensor(norm)