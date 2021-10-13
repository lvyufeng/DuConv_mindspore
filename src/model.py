import mindspore.nn as nn
import mindspore.ops as P
from .gru import GRU
from .bert import BertModel, MaskedFill
from .dropout import Dropout
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import constexpr
import mindspore.common.dtype as mstype

class MemoryNet(nn.Cell):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.kn_embedding = nn.Embedding(vocab_size, embed_size, embedding_table='xavier_uniform')
        self.gru = GRU(embed_size, hidden_size, candidate_activation='relu')
        self.fc = nn.Dense(hidden_size * 2, 256, weight_init='xavier_uniform', has_bias=False)

    def construct(self, inputs, seq_length):
        embed_inputs = self.kn_embedding(inputs)
        outputs = self.gru(embed_inputs, seq_length=seq_length)
        proj_outputs = self.fc(outputs)
        return outputs, proj_outputs

class Attention(nn.Cell):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Dense(hidden_size, 1, weight_init='xavier_uniform', has_bias=False)
        self.masked_fill = MaskedFill(-1e12)
        self.equal = P.Equal()

    def construct(self, hidden_mem, encoder_vec, encoder_vec_proj, seq_length):
        mask = self.make_mask(encoder_vec.shape[:2], mstype.int32, 1, P.ExpandDims()(seq_length, 1))
        mask = P.ExpandDims()(mask, 2)
        mask_broadcast = P.Cast()(P.BroadcastTo((encoder_vec_proj.shape))(P.Cast()(mask, mstype.int32)), mstype.bool_)
        # print(encoder_vec, mask)
        # print(encoder_vec_proj.shape, encoder_vec.shape)
        hidden_mem = P.ExpandDims()(hidden_mem, 1)
        concated = P.BroadcastTo((encoder_vec_proj.shape))(hidden_mem)
        concated = encoder_vec_proj + concated
        concated = P.Tanh()(concated)
        concated = self.masked_fill(concated, mask_broadcast)
        # print(concated)
        attention_weights = self.fc(concated)
        attention_weights = self.masked_fill(attention_weights, mask)
        attention_weights = P.Softmax(1)(attention_weights)
        # print(encoder_vec, attention_weights, attention_weights.shape, encoder_vec.shape)
        scaled = attention_weights * encoder_vec
        context = P.ReduceSum(False)(scaled, 1)
        return context

    def make_mask(self, shape, dtype, range_dim, seq_length):
        output = P.Ones()(shape, mstype.float32)
        output = P.CumSum()(output, range_dim)
        output = P.Cast()(output, dtype)
        mask = P.Greater()(output, seq_length)
        return mask
class Retrieval(nn.Cell):
    def __init__(self, config, use_kn=False):
        super().__init__()
        self.use_kn = use_kn
        self.bert = BertModel(config)
        self.memory = MemoryNet(config.vocab_size, config.hidden_size, 128)
        self.attention = Attention(config.hidden_size)
        self.fc = nn.Dense(config.hidden_size * 2 if self.use_kn else config.hidden_size, 2, weight_init=TruncatedNormal(config.initializer_range))
        self.dropout = Dropout(1-config.hidden_dropout_prob)

    def construct(self, input_ids, segment_ids, position_ids=None, kn_ids=None, seq_length=None):
        # print(kn_ids)
        if len(seq_length.shape) != 1:
            seq_length = P.Squeeze(1)(seq_length)
        _, h_pooled = self.bert(input_ids, segment_ids, position_ids)
        if self.use_kn:
            memory_outputs, memory_proj_outputs = self.memory(kn_ids, seq_length)
            kn_context = self.attention(h_pooled, memory_outputs, memory_proj_outputs, seq_length)
            cls_feats = P.Concat(1)((h_pooled, kn_context))
        else:
            cls_feats = h_pooled
        cls_feats = self.dropout(cls_feats)
        logits = self.fc(cls_feats)
        return logits

class RetrievalWithLoss(nn.Cell):
    def __init__(self, config, use_kn):
        super().__init__()
        self.network = Retrieval(config, use_kn)
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.squeeze = P.Squeeze(1)
    
    def construct(self, *inputs):
        # print(inputs[-1].shape)
        out = self.network(*inputs[:-1])
        # print(out.shape, inputs[-1].shape)
        labels = self.squeeze(inputs[-1])
        return self.loss(out, labels)

class RetrievalWithSoftmax(nn.Cell):
    def __init__(self, config, use_kn):
        super().__init__()
        self.network = Retrieval(config, use_kn)
        self.softmax = nn.Softmax()
    
    def construct(self, *inputs):
        out = self.network(*inputs)
        out = self.softmax(out)
        return out
