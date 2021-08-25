import math
import numpy as np
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.ops.primitive import constexpr

activation_map = {
    'relu': P.ReLU(),
    'sigmoid': P.Sigmoid(),
    'tanh': P.Tanh()
}

class GRUCell(nn.Cell):
    def __init__(self, gate_activation='sigmoid', candidate_activation='tanh', origin_mode=False):
        super().__init__()
        self.gate_act = activation_map.get(gate_activation, None)
        self.cand_act = activation_map.get(candidate_activation, None)
        if self.gate_act is None or self.cand_act is None:
            raise ValueError('Unsupported activation function.')
        self.split = P.Split(1, 3)
        self.matmul = P.MatMul(False, True)
        self.origin_mode = origin_mode

    def construct(self, inputs, hidden, w_ih, w_hh, b_ih, b_hh):
        '''GRU cell function'''
        if b_ih is None:
            gi = self.matmul(inputs, w_ih)
            gh = self.matmul(hidden, w_hh)
        else:
            gi = self.matmul(inputs, w_ih) + b_ih
            gh = self.matmul(hidden, w_hh) + b_hh
        i_r, i_i, i_n = self.split(gi)
        h_r, h_i, h_n = self.split(gh)

        resetgate = self.gate_act(i_r + h_r)
        inputgate = self.gate_act(i_i + h_i)
        newgate = self.cand_act(i_n + resetgate * h_n)
        if self.origin_mode:
            hy = newgate + inputgate * (hidden - newgate)
        else:
            hy = hidden + inputgate * (newgate - hidden)
        return hy

class DynamicGRU(nn.Cell):
    def __init__(self, gate_activation='sigmoid', candidate_activation='tanh', origin_mode=False):
        super().__init__()
        self.cell = GRUCell(gate_activation, candidate_activation, origin_mode)
    
    def construct(self, x, h_0, w_ih, w_hh, b_ih, b_hh):
        '''recurrent steps without sequence length'''
        time_step = x.shape[0]
        outputs = []
        t = 0
        h = h_0
        while t < time_step:
            x_t = x[t:t+1:1]
            x_t = P.Squeeze(0)(x_t)
            h = self.cell(x_t, h, w_ih, w_hh, b_ih, b_hh)
            if self.is_lstm:
                outputs.append(h[0])
            else:
                outputs.append(h)
            t += 1
        outputs = P.Stack()(outputs)
        return outputs, h

@constexpr
def _init_state(shape, dtype, is_lstm):
    hx = Tensor(np.zeros(shape), dtype)
    cx = Tensor(np.zeros(shape), dtype)
    if is_lstm:
        return (hx, cx)
    return hx

class GRU(nn.Cell):
    def __init__(self, input_size, hidden_size, gate_activation='sigmoid', candidate_activation='tanh', origin_mode=False):
        super().__init__()
        num_directions = 2
        gate_size = hidden_size * 3
        self.hidden_size = hidden_size

        self.w_ih_list = []
        self.w_hh_list = []
        self.b_ih_list = []
        self.b_hh_list = []
        stdv = 1 / math.sqrt(self.hidden_size)
        for direction in range(num_directions):
            suffix = '_reverse' if direction == 1 else ''

            self.w_ih_list.append(Parameter(
                Tensor(np.random.uniform(-stdv, stdv, (gate_size, input_size)).astype(np.float32)),
                name='weight_ih_l{}'.format(suffix)))
            self.w_hh_list.append(Parameter(
                Tensor(np.random.uniform(-stdv, stdv, (gate_size, hidden_size)).astype(np.float32)),
                name='weight_hh_l{}'.format(suffix)))
            self.b_ih_list.append(Parameter(
                Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                name='bias_ih_l{}'.format(suffix)))
            self.b_hh_list.append(Parameter(
                Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                    name='bias_hh_l{}'.format(suffix)))
        self.w_ih_list = ParameterTuple(self.w_ih_list)
        self.w_hh_list = ParameterTuple(self.w_hh_list)
        self.b_ih_list = ParameterTuple(self.b_ih_list)
        self.b_hh_list = ParameterTuple(self.b_hh_list)

        self.gru = DynamicGRU(gate_activation, candidate_activation, origin_mode)
        self.reverse = P.ReverseV2([0])
        self.reverse_sequence = P.ReverseSequence(0, 1)

    def construct(self, x, hx=None, seq_length=None):
        max_batch_size = x.shape[0]
        if hx is None:
            hx = _init_state((2, max_batch_size, self.hidden_size),
                             x.dtype)
        w_f_ih, w_f_hh, b_f_ih, b_f_hh = \
            self.w_ih_list[0], self.w_hh_list[0], \
            self.b_ih_list[0], self.b_hh_list[0]
        w_b_ih, w_b_hh, b_b_ih, b_b_hh = \
            self.w_ih_list[1], self.w_hh_list[1], \
            self.b_ih_list[1], self.b_hh_list[1]
        h_f_i = hx[0]
        h_b_i = hx[1]
        if seq_length is None:
            x_b = self.reverse(x)
        else:
            x_b = self.reverse_sequence(x, seq_length)
        output_f, h_t_f = self.rnn(x, h_f_i, seq_length, w_f_ih, w_f_hh, b_f_ih, b_f_hh)
        output_b, h_t_b = self.rnn(x_b, h_b_i, seq_length, w_b_ih, w_b_hh, b_b_ih, b_b_hh)
        if seq_length is None:
            output_b = self.reverse(output_b)
        else:
            output_b = self.reverse_sequence(output_b, seq_length)
        output = P.Concat(2)((output_f, output_b))
        return output
