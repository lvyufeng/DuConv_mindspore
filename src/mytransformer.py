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

class FeedForward(nn.Cell):
    """
    Apply two-layer feed forward
    """
    def __init__(self,
                 in_channels,#in_channels=hidden_size256
                 hidden_size,#hidden_size = intermediate_size256*4
                 out_channels,#out_channels=hidden_size = d_model =256
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
                              out_channels,#==d_model
                              weight_init=weight_variable([out_channels, hidden_size])).to_float(compute_type)

        self.preprocess = LayerPreprocess(in_channels=in_channels)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob,in_channels=in_channels)

        self.reshape = P.Reshape()
        self.shape = (-1, in_channels)
        self.dropout = nn.Dropout(1 - hidden_dropout_prob)
        self.use_dropout = hidden_dropout_prob > 0

    def construct(self, input_tensor):
        #input_tensor = self.reshape(input_tensor, self.shape)
        output = self.preprocess(input_tensor+input_tensor)
        output = self.conv1(output)
        if self.use_dropout:
            output = self.dropout(output)
        output = self.conv2(output)
        output = self.postprocess(output, input_tensor)
        return output

class LayerPostprocess(nn.Cell):
    """
    postprocess output of each layer.
    """
    def __init__(self,
                 in_channels,dropout_prob=0.1):
        super(LayerPostprocess, self).__init__()
        self.add = P.Add()
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.use_dropout = dropout_prob > 0
        self.layernorm = nn.LayerNorm((in_channels,))

    def construct(self, hidden_tensor,input_tensor):
        output = hidden_tensor
        if self.use_dropout:
            output = self.dropout(output)
        output = self.add(output, input_tensor)
        output = self.layernorm(output)
        return output

class MultiheadAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".
    """
    def __init__(self,
                 batch_size,
                 from_tensor_width,#hidden_size
                 to_tensor_width,#hidden_size
                 out_tensor_width,#hidden_size
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
                 do_return_2d_tensor=False,
                 compute_type=mstype.float16):
        super(MultiheadAttention, self).__init__()
        self.batch_size = batch_size
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head#hidden_size / num_attention_heads
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
                                    has_bias=True,
                                    weight_init=weight_variable([units, from_tensor_width])).to_float(compute_type)
        self.key_layer = nn.Dense(to_tensor_width,
                                  units,
                                  has_bias=True,
                                  weight_init=weight_variable([units, to_tensor_width])).to_float(compute_type)
        self.value_layer = nn.Dense(to_tensor_width,
                                    units,
                                    has_bias=True,
                                    weight_init=weight_variable([units, to_tensor_width])).to_float(compute_type)
        self.out_layer = nn.Dense(units,
                                  out_tensor_width,
                                  has_bias=True,
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

       
        self.softmax_cast = P.Cast()
        #self.stack = P.Stack(axis=1)

    def construct(self, from_tensor, to_tensor, seq_length, enc_seq_length, attention_mask=None):
        """Apply multihead attention.""" #from_tensor = to_tensor = output  seq_length = enc_seq_length
        from_seq_length = seq_length
        to_seq_length = enc_seq_length
        shape_from = (self.batch_size, from_seq_length, self.num_attention_heads, self.size_per_head)
        shape_to = (self.batch_size, to_seq_length, self.num_attention_heads, self.size_per_head)
        #shape_from = shape_to
        if self.do_return_2d_tensor:#128*256,8*hidden_size / num_attention_heads
            shape_return = (self.batch_size * from_seq_length, self.num_attention_heads * self.size_per_head)
            if from_seq_length == -1:
                shape_return = (-1, self.num_attention_heads * self.size_per_head)
        else:
            shape_return = (self.batch_size, from_seq_length, self.num_attention_heads * self.size_per_head)

        # reshape 2d/3d input tensors to 2d   那边是三维的送进去
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)
        #print("query_out",query_out.shape)
        #print("shape_from",shape_from)
        query_layer = self.reshape(query_out, shape_from)#shape_from：[128,256,8,256/8]-->128*256*256
        query_layer = self.transpose(query_layer, self.trans_shape)#[128,8,256,256/8]
        key_layer = self.reshape(key_out, shape_to)
        key_layer = self.transpose(key_layer, self.trans_shape)#[128,8,256,256/8]
        value_layer = self.reshape(value_out, shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)

        scale = self.size_per_head ** -0.5
        key_layer = key_layer * scale
        attention_scores = self.matmul_trans_b(query_layer, key_layer)
        attention_scores = self.multiply(attention_scores, self.scores_mul)

        if self.has_attention_mask:
            attention_scores = attention_scores + attention_mask
            #attention_mask = self.expand_dims(attention_mask, 1)
            # multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
            #                         self.cast(attention_mask, self.get_dtype(attention_scores)))
            # adder = self.multiply(multiply_out, self.multiply_data)
            # attention_scores = self.add(adder, attention_scores)

        attention_scores = self.softmax_cast(attention_scores, mstype.float16)#只是转换类型
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.softmax_cast(attention_probs, mstype.float16)
        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)
        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.transpose(context_layer, self.trans_shape)#trans_shape: [0,2,1,3]
        context_layer = self.reshape(context_layer, shape_return)#由[128,256,8,256/8]-->[128,256,256]
        context_layer = self.out_layer(context_layer)#返回 [128,256,256]
        return context_layer


class LayerPreprocess(nn.Cell):
    """
    preprocess input of each layer.
    """
    def __init__(self,
                 in_channels=None):
        super(LayerPreprocess, self).__init__()
        self.layernorm = nn.LayerNorm((in_channels,))
        self.cast = P.Cast()
        self.get_dtype = P.DType()

    def construct(self, input_tensor):
        output = self.cast(input_tensor, mstype.float16)#送进来时已经加了 input_tensor+input_tensor
        output = self.layernorm(output)
        output = self.cast(output, self.get_dtype(input_tensor))
        return output

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
        self.size_per_head = int(hidden_size / num_attention_heads)#emb_size*4/8
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
            do_return_2d_tensor=False,
            compute_type=compute_type)

        self.preprocess = LayerPreprocess(in_channels=hidden_size)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob,in_channels = hidden_size)

        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)
    def construct(self, input_tensor, memory_tensor, attention_mask, seq_length, enc_seq_length):
        """Apply self-attention.""" #enc_seq_length = seq_length   input_tensor = memory_tensor
        orgin_input_tensor = input_tensor
        input_tensor = self.reshape(input_tensor, self.shape)
        memory_tensor = self.reshape(memory_tensor, self.shape)

        output = self.preprocess(input_tensor + input_tensor)

        if not self.is_encdec_att:
            memory_tensor = output

        attention_output = self.attention(output, memory_tensor, seq_length, enc_seq_length, attention_mask)#[]
        #print("attention_output",attention_output.shape)
        #print("orgin_input_tensor",orgin_input_tensor.shape)
        output = self.postprocess(attention_output, orgin_input_tensor)#加dropout [128,256,256]
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
    def __init__(self, config, is_training):
        super(TransformerEncoder, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        self.batch_size = config.batch_size##n ,d,d,d,d,pd,ad,rd,ha,prc,poc
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.use_one_hot_embeddings = config.use_one_hot_embeddings
        self.initializer_range = config.initializer_range
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.hidden_act = config.hidden_act
        self.compute_type = config.compute_type

        layers = []
        for _ in range(self.num_hidden_layers):
            layer = EncoderCell(batch_size=self.batch_size,
                                hidden_size=self.hidden_size,
                                num_attention_heads=self.num_attention_heads,
                                intermediate_size=self.intermediate_size,
                                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                                use_one_hot_embeddings=self.use_one_hot_embeddings,
                                initializer_range=self.initializer_range,
                                hidden_dropout_prob=self.hidden_dropout_prob,
                                hidden_act=self.hidden_act,
                                compute_type=self.compute_type)
            layers.append(layer)
        self.layers = nn.CellList(layers)

        self.layer_preprocess = LayerPreprocess(in_channels=self.hidden_size)

        # self.reshape = P.Reshape()
        # self.shape = (-1, self.hidden_size)

    def construct(self, input_tensor, attention_mask, seq_length):
        """Apply encoder."""
        out_shape = (self.batch_size, seq_length, self.hidden_size)
        #prev_output = self.reshape(input_tensor, self.shape)
        prev_output = input_tensor

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, seq_length)
            prev_output = layer_output

        prev_output = self.layer_preprocess(prev_output+prev_output)
        #output = self.reshape(prev_output, out_shape)
        return prev_output