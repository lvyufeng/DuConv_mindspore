import math
import copy
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops.primitive import constexpr
from weight_init import normal_weight, weight_variable

class TransformerConfig:
    """
    Configuration for `Transformer`.
    """
    def __init__(self,
                 batch_size,
                 emb_size=256,
                 gru_hidden_size=128,
                 num_directions=True,
                 seq_length=256,
                 hidden_size=1024,
                 num_hidden_layers=12,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 hidden_act="relu",
                 hidden_dropout_prob=0.3,
                 attention_probs_dropout_prob=0.3,
                 max_position_embeddings=256,
                 initializer_range=0.02,
                 label_smoothing=0.1,
                 length_penalty_weight=1.0,
                 dtype=mstype.float16,
                 compute_type=mstype.float16):
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.gru_hidden_size = gru_hidden_size
        self.num_directions = num_directions
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.label_smoothing = label_smoothing
        self.length_penalty_weight = length_penalty_weight
        self.dtype = dtype
        self.compute_type = compute_type


class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.
    """
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(normal_weight([vocab_size, embedding_size], embedding_size))
        self.expand = P.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float16)
        self.off_value = Tensor(0.0, mstype.float16)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, input_ids):
        """Get a embeddings lookup table with a fixed dictionary and size."""
        input_shape = self.shape(input_ids)

        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        out_shape = input_shape + (self.embedding_size,)
        output = self.reshape(output_for_reshape, out_shape)
        return output, self.embedding_table


def position_encoding(length,
                      depth,
                      min_timescale=1,
                      max_timescale=1e4):
    """
    Create Tensor of sinusoids of different frequencies.
    """
    depth = depth // 2
    positions = np.arange(length, dtype=np.float16)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float16) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional embeddings to word embeddings.
    """
    def __init__(self,
                 embedding_size,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 max_position_embeddings=256,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()
        self.scores_mul = Tensor([math.sqrt(float(embedding_size))], dtype=mstype.float16)
        self.multiply = P.Mul()
        self.add = P.Add()
        self.dropout = nn.Dropout(1 - dropout_prob, dtype=mstype.float16)
        self.use_dropout = dropout_prob > 0
        self.expand_dims = P.ExpandDims()
        self.position_embedding_table = Tensor(position_encoding(max_position_embeddings, embedding_size),
                                               mstype.float16)
        self.shape = P.Shape()

    def construct(self, word_embeddings):
        """Postprocessors apply positional embeddings to word embeddings."""
        input_shape = self.shape(word_embeddings)
        input_len = input_shape[1]

        output = self.multiply(word_embeddings, self.scores_mul)

        # add position embeddings
        position_embeddings = self.position_embedding_table[0:input_len:1, ::]
        position_embeddings = self.expand_dims(position_embeddings, 0)
        output = self.add(output, position_embeddings)

        if self.use_dropout:
            output = self.dropout(output)
        return output


class CastWrapper(nn.Cell):
    """
    Cast wrapper.
    """
    def __init__(self, src_type=mstype.float16, dst_type=mstype.float16):
        super(CastWrapper, self).__init__()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        return self.cast(x, self.dst_type)

class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Parameter(np.ones(normalized_shape, dtype=np.float32))
        self.beta = Parameter(np.zeros(normalized_shape, dtype=np.float32))
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.power = P.Pow()
        self.shape = P.Shape()

    def construct(self, x):
        mean = self.reduce_mean(x, -1)
        diff = x - mean
        var = self.reduce_mean(self.power(diff, 2), -1)
        normalized_x = (x - mean) / ((var + self.eps) ** 0.5)
        output = self.gamma * normalized_x + self.beta
        return output

class LayerPreprocess(nn.Cell):
    """
    preprocess input of each layer.
    """
    def __init__(self,
                 in_channels=None):
        super(LayerPreprocess, self).__init__()
        #self.layernorm = nn.LayerNorm((in_channels,))
        self.layernorm = LayerNorm((in_channels,))
        self.cast = P.Cast()
        self.get_dtype = P.DType()

    def construct(self, input_tensor):
        output = self.cast(input_tensor, mstype.float16)
        output = self.layernorm(output)
        output = self.cast(output, self.get_dtype(input_tensor))
        return output


class LayerPostprocess(nn.Cell):
    """
    postprocess output of each layer.
    """
    def __init__(self,
                 dropout_prob=0.1):
        super(LayerPostprocess, self).__init__()
        self.add = P.Add()
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.use_dropout = dropout_prob > 0

    def construct(self, hidden_tensor, input_tensor):
        output = hidden_tensor
        if self.use_dropout:
            output = self.dropout(output)
        output = self.add(output, input_tensor)
        return output


class MultiheadAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".
    """
    def __init__(self,
                 batch_size,
                 from_tensor_width,
                 to_tensor_width,
                 out_tensor_width,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 out_act=None,
                 has_attention_mask=True,
                 attention_probs_dropout_prob=0.0,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 do_return_2d_tensor=True,
                 compute_type=mstype.float16):
        super(MultiheadAttention, self).__init__()
        self.batch_size = batch_size
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = has_attention_mask
        assert has_attention_mask
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor

        self.scores_mul = Tensor([1.0 / math.sqrt(float(self.size_per_head))], dtype=compute_type)
        self.reshape = P.Reshape()
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)
        units = num_attention_heads * size_per_head
        self.query_layer = nn.Dense(from_tensor_width,
                                    units,
                                    has_bias=False,
                                    weight_init=weight_variable([units, from_tensor_width])).to_float(compute_type)
        self.key_layer = nn.Dense(to_tensor_width,
                                  units,
                                  has_bias=False,
                                  weight_init=weight_variable([units, to_tensor_width])).to_float(compute_type)
        self.value_layer = nn.Dense(to_tensor_width,
                                    units,
                                    has_bias=False,
                                    weight_init=weight_variable([units, to_tensor_width])).to_float(compute_type)
        self.out_layer = nn.Dense(units,
                                  out_tensor_width,
                                  has_bias=False,
                                  weight_init=weight_variable([out_tensor_width, units])).to_float(compute_type)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = Tensor([-10000.0,], dtype=compute_type)
        self.batch_num = batch_size * num_attention_heads
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1 - attention_probs_dropout_prob)
        self.use_dropout = attention_probs_dropout_prob > 0

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()

        self.cast_compute_type = CastWrapper(dst_type=compute_type)
        self.softmax_cast = P.Cast()
        #self.stack = P.Stack(axis=1)

    def construct(self, from_tensor, to_tensor, seq_length, enc_seq_length, attention_mask=None):
        """Apply multihead attention."""
        from_seq_length = seq_length
        to_seq_length = enc_seq_length
        shape_from = (self.batch_size, from_seq_length, self.num_attention_heads, self.size_per_head)
        shape_to = (self.batch_size, to_seq_length, self.num_attention_heads, self.size_per_head)
        if self.do_return_2d_tensor:
            shape_return = (self.batch_size * from_seq_length, self.num_attention_heads * self.size_per_head)
            if from_seq_length == -1:
                shape_return = (-1, self.num_attention_heads * self.size_per_head)
        else:
            shape_return = (self.batch_size, from_seq_length, self.num_attention_heads * self.size_per_head)

        # reshape 2d/3d input tensors to 2d
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)

        query_layer = self.reshape(query_out, shape_from)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, shape_to)
        key_layer = self.transpose(key_layer, self.trans_shape)

        attention_scores = self.matmul_trans_b(query_layer, key_layer)
        attention_scores = self.multiply(attention_scores, self.scores_mul)

        if self.has_attention_mask:
            # attention_scores = self.add(attention_scores, attention_mask)
            # attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))
            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        attention_scores = self.softmax_cast(attention_scores, mstype.float16)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.softmax_cast(attention_probs, self.get_dtype(key_layer))
        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, shape_return)
        context_layer = self.out_layer(context_layer)
        return context_layer


class SelfAttention(nn.Cell):
    """
    Apply self-attention.
    """
    def __init__(self,
                 batch_size,
                 hidden_size,
                 num_attention_heads=16,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 has_attention_mask=True,
                 is_encdec_att=False,
                 compute_type=mstype.float16):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))
        self.size_per_head = int(hidden_size / num_attention_heads)
        self.is_encdec_att = is_encdec_att

        self.attention = MultiheadAttention(
            batch_size=batch_size,
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            out_tensor_width=hidden_size,
            num_attention_heads=num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=True,
            compute_type=compute_type)

        self.preprocess = LayerPreprocess(in_channels=hidden_size)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)
    def construct(self, input_tensor, memory_tensor, attention_mask, seq_length, enc_seq_length):
        """Apply self-attention."""
        input_tensor = self.reshape(input_tensor, self.shape)
        memory_tensor = self.reshape(memory_tensor, self.shape)

        output = self.preprocess(input_tensor)

        if not self.is_encdec_att:
            memory_tensor = output

        attention_output = self.attention(output, memory_tensor, seq_length, enc_seq_length, attention_mask)
        output = self.postprocess(attention_output, input_tensor)
        return output


class FeedForward(nn.Cell):
    """
    Apply two-layer feed forward
    """
    def __init__(self,
                 in_channels,
                 hidden_size,
                 out_channels,
                 hidden_act="relu",
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 compute_type=mstype.float16):
        super(FeedForward, self).__init__()

        self.conv1 = nn.Dense(in_channels,
                              hidden_size,
                              activation=hidden_act,
                              weight_init=weight_variable([hidden_size, in_channels])).to_float(compute_type)
        self.conv2 = nn.Dense(hidden_size,
                              out_channels,
                              weight_init=weight_variable([out_channels, hidden_size])).to_float(compute_type)

        self.preprocess = LayerPreprocess(in_channels=in_channels)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

        self.reshape = P.Reshape()
        self.shape = (-1, in_channels)
        self.dropout = nn.Dropout(1 - hidden_dropout_prob)
        self.use_dropout = hidden_dropout_prob > 0

    def construct(self, input_tensor):
        input_tensor = self.reshape(input_tensor, self.shape)
        output = self.preprocess(input_tensor)
        output = self.conv1(output)
        if self.use_dropout:
            output = self.dropout(output)
        output = self.conv2(output)
        output = self.postprocess(output, input_tensor)
        return output


class EncoderCell(nn.Cell):
    """
    Encoder cells used in Transformer.
    """
    def __init__(self,
                 batch_size,
                 hidden_size=1024,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float16):
        super(EncoderCell, self).__init__()
        self.attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            is_encdec_att=False,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)

    def construct(self, hidden_states, attention_mask, seq_length):
        # self-attention with ln, res
        attention_output = self.attention(hidden_states, hidden_states, attention_mask, seq_length, seq_length)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output


class TransformerEncoder(nn.Cell):
    """
    Multi-layer transformer encoder.
    """
    def __init__(self,
                 batch_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float16):
        super(TransformerEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        layers = []
        for _ in range(num_hidden_layers):
            layer = EncoderCell(batch_size=batch_size,
                                hidden_size=hidden_size,
                                num_attention_heads=num_attention_heads,
                                intermediate_size=intermediate_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                use_one_hot_embeddings=use_one_hot_embeddings,
                                initializer_range=initializer_range,
                                hidden_dropout_prob=hidden_dropout_prob,
                                hidden_act=hidden_act,
                                compute_type=compute_type)
            layers.append(layer)
        self.layers = nn.CellList(layers)

        self.layer_preprocess = LayerPreprocess(in_channels=hidden_size)

        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, attention_mask, seq_length):
        """Apply encoder."""
        out_shape = (self.batch_size, seq_length, self.hidden_size)
        prev_output = self.reshape(input_tensor, self.shape)

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, seq_length)
            prev_output = layer_output

        prev_output = self.layer_preprocess(prev_output)
        output = self.reshape(prev_output, out_shape)
        return output

class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.
    """
    def __init__(self):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.input_mask = None

        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = (-1, 1, 256)#(-1, 1, config.seq_length)

    def construct(self, input_mask):
        """Create attention mask according to input mask."""
        # attention_mask = self.cast(self.reshape(input_mask, self.shape), mstype.float16)
        attention_mask = self.cast(input_mask, mstype.float16)
        return attention_mask


# class CreateAttentionMaskFromInputMask(nn.Cell):
#     """
#     Create attention mask according to input mask.
#     """
#     def __init__(self):
#         super(CreateAttentionMaskFromInputMask, self).__init__()
#         self.cast = P.Cast()
#         self.reshape = P.Reshape()
#         self.shape = P.Shape()
#         self.batch_matmul = P.BatchMatMul()

#     def construct(self, input_mask):
#         """Create attention mask according to input mask."""
#         input_shape = self.shape(input_mask)
#         shape_right = (input_shape[0], 1, input_shape[1])
#         shape_left = input_shape + (1,)

#         input_mask = self.cast(input_mask, mstype.float16)
#         mask_left = self.reshape(input_mask, shape_left)
#         # print("input_mask:", input_mask.shape)
#         # print("shape_left:", shape_left.shape)
#         # print("shape_right:", shape_right.shape)
#         mask_right = self.reshape(input_mask, shape_right)
#         attention_mask = self.batch_matmul(mask_left, mask_right)

#         return attention_mask

class TransformerModel(nn.Cell):
    """
    Transformer with encoder and decoder.
    """
    def __init__(self,
                 config,
                 is_training,
                 vocab_size,
                 use_one_hot_embeddings=False):
        super(TransformerModel, self).__init__()
        config = copy.deepcopy(config)
        self.is_training = is_training
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size
        self.seq_length = config.seq_length
        self.last_idx = self.num_hidden_layers - 1

        # self.id_tfm_embedding_lookup = EmbeddingLookup(
        #     vocab_size=vocab_size,
        #     embedding_size=self.embedding_size,
        #     use_one_hot_embeddings=use_one_hot_embeddings,
        #     initializer_range=config.initializer_range)
        # self.pos_tfm_embedding_lookup = EmbeddingLookup(
        #     vocab_size=config.max_position_embeddings,
        #     embedding_size=self.embedding_size,
        #     use_one_hot_embeddings=use_one_hot_embeddings,
        #     initializer_range=config.initializer_range)
        # self.seg_tfm_embedding_lookup = EmbeddingLookup(
        #     vocab_size=2,
        #     embedding_size=self.embedding_size,
        #     use_one_hot_embeddings=use_one_hot_embeddings,
        #     initializer_range=config.initializer_range)
        self.tfm_embedding_postprocessor_for_encoder = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=0.02,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
        self.tfm_encoder = TransformerEncoder(
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=config.initializer_range,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_act=config.hidden_act,
            compute_type=config.compute_type)

        self.cast = P.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = CastWrapper(dst_type=config.compute_type)
        self.expand = P.ExpandDims()
        self.multiply = P.Mul()
        self.shape = P.Shape()
        self.add = P.Add()
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask()

    def construct(self, context_emb_out, context_position_emb_out, context_segment_emb_out, source_mask):
        """Transformer with encoder""" #context_ids, context_pos_ids, context_segment_ids
        #print("context_ids",context_ids.shape)
        seq_length = self.seq_length#. self.shape(context_ids)[1]#length

        # process source sentence
        # context_emb_out, embedding_tables_1 = self.id_tfm_embedding_lookup(context_ids)
        # context_position_emb_out, embedding_tables_2 = self.pos_tfm_embedding_lookup(context_pos_ids)
        # context_segment_emb_out, embedding_tables_3 = self.seg_tfm_embedding_lookup(context_segment_ids)
        context_emb_out = context_emb_out +  context_position_emb_out
        context_emb_out = context_emb_out + context_segment_emb_out
        src_word_embeddings = context_emb_out
        src_embedding_output = self.tfm_embedding_postprocessor_for_encoder(src_word_embeddings)
        # attention mask [batch_size, seq_length, seq_length]
        enc_attention_mask = self._create_attention_mask_from_input_mask(source_mask)
        # transformer encoder
        encoder_output = self.tfm_encoder(self.cast_compute_type(src_embedding_output),
                                          self.cast_compute_type(enc_attention_mask),
                                          seq_length)
        return encoder_output