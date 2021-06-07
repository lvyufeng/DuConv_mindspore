import numpy as np
import copy
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ops
from mindspore.common.initializer import TruncatedNormal,Constant
from transformer import TransformerModel
from src.rnns import GRU
"""
Retrieval model
"""
class Attention(nn.Cell):
    def __init__(self):
        super(Attention,self).__init__()
        self.fc = nn.Dense(256,1)
        #self.matmul = ops.BatchMatMul()
        self.matmul = nn.MatMul().to_float(mindspore.float16)
        self.transpose = ops.Transpose()
        self.expand_dims = ops.ExpandDims()
        self.squeeze_context = ops.Squeeze(1)
        self.squeeze_weights = ops.Squeeze(-1)
        self.add = ops.Add()
        self.tanh = ops.Tanh()
        self.softmax = ops.Softmax()
        self.cast = ops.Cast()

    def construct(self, hidden_mem, encoder_vec, encoder_vec_proj):        
        hidden_mem = self.expand_dims(hidden_mem,1)
        concated = ops.BroadcastTo((encoder_vec_proj.shape[0], encoder_vec_proj.shape[1],256))(hidden_mem)           
        concated = self.add(encoder_vec_proj, concated)       
        concated = self.tanh(concated)
        attention_weights = self.fc(concated)   
        attention_weights = self.squeeze_weights(attention_weights)     
        attention_weights = self.softmax(attention_weights)   
        att_context=[]
        attention_weights = self.expand_dims(attention_weights, 1)
        attention_weights = self.cast(attention_weights, mindspore.float16)
        encoder_vec = self.cast(encoder_vec, mindspore.float16)
        context = self.matmul(attention_weights, encoder_vec)
        context = self.cast(context, mindspore.float32)
        context = self.squeeze_context(context)
        return context

class RetrievalModel(nn.Cell):
    def __init__(self,
                 config,
                 voc_size,
                 num_labels,
                 is_training):
        super(RetrievalModel, self).__init__()  
        config = copy.deepcopy(config)
        self.seq_length = config.seq_length
        self.is_training = is_training
        self._emb_size = config.emb_size
        self._voc_size = voc_size
        self.sent_types = 2
        self.max_position_embeddings = config.max_position_embeddings
        self.attention = Attention()
        self.batch_size = config.batch_size
        self.maxlen_gru=config.seq_length
        self.gru = GRU(input_size=self._emb_size, hidden_size=config.gru_hidden_size, bidirectional=True)
        self.dense = nn.Dense(config.seq_length,self._emb_size)
        #self.gru_embed_input = nn.Embedding(voc_size, self._emb_size)  
        # self.encoder = TransformerEncoder(config,is_training)#mytransformer
        self.encoder = TransformerModel(config,self.is_training,voc_size,use_one_hot_embeddings=False)
        # self.encoder = Encoder(voc_size, d_model, d_k, n_heads, d_ff, n_layers, src_len)       
        self.reshape = ops.Reshape()
        self.gather = ops.Gather()
        self.concat_trans_gru = ops.Concat(1)
        self.cast = ops.Cast()  
        self.trans_weight_init = Parameter(Tensor(np.random.normal(0.0, 0.02,(self._emb_size, self._emb_size)).astype(np.float32)))
        self.trans_fc = nn.Dense(self._emb_size,self._emb_size,weight_init=self.trans_weight_init,activation="tanh")
        weight_init_ = TruncatedNormal(0.02)
        bias_init_ = Constant(0.)
        self.logit_fc = nn.Dense(2*self._emb_size, num_labels, weight_init=weight_init_, bias_init=bias_init_)
        self.trans_logit_fc = nn.Dense(self._emb_size, num_labels)
        self.dropout = nn.Dropout(0.9)
        self.testdropout = nn.Dropout(1.0)
        self.cast = ops.Cast()
        self.embedding = nn.Embedding(self._voc_size, self._emb_size)
        self.posembedding = nn.Embedding(self.max_position_embeddings, self._emb_size)
        self.segembedding = nn.Embedding(self.sent_types, self._emb_size)
        self.transpose = ops.Transpose()

    def construct(self,kn_ids, context_next_sent_index,context_ids, context_pos_ids, context_segment_ids, context_attn_mask, seq_lengths):
        context_emb_out = self.embedding(context_ids)
        context_position_emb_out = self.posembedding(context_pos_ids)
        context_segment_emb_out = self.segembedding(context_segment_ids)
        context_emb_out = context_emb_out +  context_position_emb_out
        context_emb_out = context_emb_out + context_segment_emb_out

        context_enc_out = self.encoder(context_emb_out, context_emb_out, context_emb_out, context_attn_mask)
        context_enc_out = self.cast(context_enc_out, mindspore.float32)

        reshaped_emb_out = self.reshape(context_enc_out, (-1, self._emb_size))
        next_sent_index = self.cast(context_next_sent_index, mindspore.int32)
        next_sent_feat = self.gather(reshaped_emb_out, next_sent_index,0)
        next_sent_feat = self.trans_fc(next_sent_feat) #(2,256)                       
        #kn_emb_out = self.gru_embed_input(kn_ids)
        x = self.embedding(kn_ids)
        x = self.transpose(x, (1, 0, 2))
        memory_encoder_out,_ = self.gru(x, seq_lengths)
        memory_encoder_out = self.transpose(memory_encoder_out, (1, 0, 2))
        # _,memory_encoder_out = self.gru(x)
        memory_encoder_out = self.cast(memory_encoder_out, mindspore.float32)
        memory_encoder_proj_out = self.dense(memory_encoder_out)
        kn_context = self.attention(next_sent_feat, memory_encoder_out, memory_encoder_proj_out)  
        cls_feats = self.concat_trans_gru((next_sent_feat, kn_context))
        if  self.is_training:
            cls_feats = self.dropout(cls_feats)
        else:
            cls_feats = self.testdropout(cls_feats)
        logits = self.logit_fc(cls_feats)
        return logits




