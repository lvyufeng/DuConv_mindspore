import mindspore
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from typing import Optional
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, XavierUniform, Zero

class GELU(nn.Cell):
    def __init__(self):
        super().__init__()
        self.erf = P.Erf()
        self.sqrt = P.Sqrt()
        self.const0 = Tensor(0.5, mindspore.float32)
        self.const1 = Tensor(1.0, mindspore.float32)
        self.const2 = Tensor(2.0, mindspore.float32)
    def construct(self, x):
        return x * self.const0 * (self.const1 + self.erf(x / self.sqrt(self.const2)))

class MaskedFill(nn.Cell):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.select = P.Select()
        self.fill = P.Fill()
        self.cast = P.Cast()
    def construct(self, inputs:Tensor, mask:Tensor):
        mask = self.cast(mask, mstype.bool_)
        masked_value = self.fill(mindspore.float32, inputs.shape, self.value)
        output = self.select(mask, masked_value, inputs)
        return output

class ScaledDotProductAttention(nn.Cell):
    def __init__(self, d_k, dropout):
        super().__init__()
        self.scale = Tensor(d_k, mindspore.float32)
        self.matmul = nn.MatMul()
        self.transpose = P.Transpose()
        self.softmax = nn.Softmax(axis=-1)
        self.sqrt = P.Sqrt()
        self.masked_fill = MaskedFill(-1e9)

        if dropout > 0.0:
            self.dropout = nn.Dropout(1-dropout)
        else:
            self.dropout = None

    def construct(self, Q, K, V, attn_mask):
        K = self.transpose(K, (0, 1, 3, 2))
        scores = self.matmul(Q, K) / self.sqrt(self.scale) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = self.masked_fill(scores, attn_mask) # Fills elements of self tensor with value where mask is one.
        # scores = scores + attn_mask
        attn = self.softmax(scores)
        context = self.matmul(attn, V)
        if self.dropout is not None:
            context = self.dropout(context)
        return context, attn

class MultiHeadAttention(nn.Cell):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.W_Q = nn.Dense(d_model, d_model)
        self.W_K = nn.Dense(d_model, d_model)
        self.W_V = nn.Dense(d_model, d_model)
        self.linear = nn.Dense(d_model, d_model)
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "embed_dim must be divisible by num_heads"
        self.layer_norm = nn.LayerNorm((d_model, ), epsilon=1e-12)
        self.attention = ScaledDotProductAttention(self.head_dim, dropout)
        # ops
        self.transpose = P.Transpose()
        self.expanddims = P.ExpandDims()
        self.tile = P.Tile()
        
    def construct(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.shape[0]
        q_s = self.W_Q(Q).view((batch_size, -1, self.n_heads, self.head_dim)) 
        k_s = self.W_K(K).view((batch_size, -1, self.n_heads, self.head_dim)) 
        v_s = self.W_V(V).view((batch_size, -1, self.n_heads, self.head_dim)) 
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.transpose(q_s, (0, 2, 1, 3)) # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.transpose(k_s, (0, 2, 1, 3)) # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.transpose(v_s, (0, 2, 1, 3)) # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = self.expanddims(attn_mask, 1)
        attn_mask = self.tile(attn_mask, (1, self.n_heads, 1, 1)) # attn_mask : [batch_size x n_heads x len_q x len_k]
        
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask)
        context = self.transpose(context, (0, 2, 1, 3)).view((batch_size, -1, self.n_heads * self.head_dim)) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context) 
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

activation_map = {
    'relu': nn.ReLU(),
    'gelu': GELU(),
}

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape

    pad_attn_mask = P.ExpandDims()(P.ZerosLike()(seq_k), 1)
    # pad_attn_mask = P.ExpandDims()(P.Equal()(seq_k, 0), 1)
    pad_attn_mask = P.Cast()(pad_attn_mask, mstype.int32)
    pad_attn_mask = P.BroadcastTo((batch_size, len_q, len_k))(pad_attn_mask)
    # pad_attn_mask = P.Cast()(pad_attn_mask, mstype.bool_)
    return pad_attn_mask

class BertConfig:
    def __init__(self,
                seq_length=128,
                vocab_size=32000,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size

class PoswiseFeedForwardNet(nn.Cell):
    def __init__(self, d_model, d_ff, activation:str='gelu'):
        super().__init__()
        self.fc1 = nn.Dense(d_model, d_ff)
        self.fc2 = nn.Dense(d_ff, d_model)
        self.activation = activation_map.get(activation, nn.GELU())
        self.layer_norm = nn.LayerNorm((d_model,), epsilon=1e-12)

    def construct(self, inputs):
        residual = inputs
        outputs = self.fc1(inputs)
        outputs = self.activation(outputs)
        
        outputs = self.fc2(outputs)
        return self.layer_norm(outputs + residual)

class BertEmbeddings(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.seg_embed = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)

        self.expand_dims = P.ExpandDims()

    def construct(self, x, seg):
        seq_len = x.shape[1]
        pos = mnp.arange(seq_len)
        pos = P.BroadcastTo(x.shape)(self.expand_dims(pos, 0))
        seg_embedding = self.seg_embed(seg)
        tok_embedding = self.tok_embed(x)
        embedding = tok_embedding + self.pos_embed(pos) + seg_embedding
        # embedding = self.tok_embed(x) + self.seg_embed(seg)
        return self.norm(embedding)

class BertEncoderLayer(nn.Cell):
    def __init__(self, d_model, n_heads, d_ff, activation, dropout):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, activation)

    def construct(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class BertEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.CellList([BertEncoderLayer(config.hidden_size, config.num_attention_heads, config.intermediate_size, config.hidden_act, config.hidden_dropout_prob) for _ in range(config.num_hidden_layers)])

    def construct(self, inputs, enc_self_attn_mask):
        outputs = inputs
        for layer in self.layers:
            outputs, enc_self_attn = layer(outputs, enc_self_attn_mask)
            # print(outputs)
        return outputs

class BertModel(nn.Cell):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = nn.Dense(config.hidden_size, config.hidden_size, activation='tanh')
        
    def construct(self, input_ids, segment_ids):
        outputs = self.embeddings(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        print(enc_self_attn_mask)
        outputs = self.encoder(outputs, enc_self_attn_mask)
        h_pooled = self.pooler(outputs[:, 0]) 
        return outputs, h_pooled