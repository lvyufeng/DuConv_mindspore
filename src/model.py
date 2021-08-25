import mindspore.nn as nn
import mindspore.ops as P
from .gru import GRU

class MemoryNet(nn.Cell):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.kn_embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = GRU(embed_size, hidden_size, candidate_activation='relu')
        self.fc = nn.Dense(hidden_size * 2, 256)

    def construct(self, inputs, seq_length):
        embed_inputs = self.kn_embedding(inputs)
        outputs = self.gru(embed_inputs, seq_length=seq_length)
        proj_outputs = self.fc(outputs)
        return outputs, proj_outputs

class Attention(nn.Cell):
    def __init__(self, auto_prefix, flags):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)

class Retrieval(nn.Cell):
    def __init__(self, auto_prefix, flags):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)
