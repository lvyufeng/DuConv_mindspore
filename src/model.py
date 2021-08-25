import mindspore.nn as nn
import mindspore.ops as P
import mindspore.common.dtype as mstype
from mindspore import Tensor, Parameter
from mindspore.common.initializer import Normal, TruncatedNormal, Constant, initializer

activation_map = {
    'relu': P.ReLU(),
    'sigmoid': P.Sigmoid(),
    'tanh': P.Tanh()
}

class GRUCell(nn.Cell):
    def __init__(self, gate_activation='sigmoid', candidate_activation='tanh'):
        super().__init__()
        self.gate_act = activation_map.get(gate_activation, None)
        self.cand_act = activation_map.get(candidate_activation, None)
        if self.gate_act is None or self.cand_act is None:
            raise ValueError('Unsupported activation function.')
        self.split = P.Split(1, 3)
        self.matmul = P.MatMul(False, True)

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
        hy = newgate + inputgate * (hidden - newgate)

        return hy

class GRU(nn.Cell):
    def __init__(self, input_size, hidden_size, gate_activation, candidate_activation):
        super().__init__()

class MemoryNet(nn.Cell):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.kn_embedding = nn.Embedding(vocab_size, embed_size)

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)