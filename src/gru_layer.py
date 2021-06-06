import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype
import mindspore
from weight_init import gru_default_state

class BidirectionGRU(nn.Cell):
    '''
    BidirectionGRU model

    Args:
        config: config of network
    '''
    def __init__(self, batch_size, encoder_embedding_size,hidden_size,max_length,is_training=True):
        super(BidirectionGRU, self).__init__()
        #if is_training:
        self.batch_size = batch_size
        self.embedding_size = encoder_embedding_size
        self.hidden_size = hidden_size
        self.weight_i, self.weight_h, self.bias_i, self.bias_h, self.init_h = gru_default_state(self.batch_size, self.embedding_size,self.hidden_size)
        self.weight_bw_i, self.weight_bw_h, self.bias_bw_i, self.bias_bw_h, self.init_bw_h = gru_default_state(self.batch_size, self.embedding_size, self.hidden_size)
        # self.reverse = P.ReverseV2(axis=[1])
        self.reverse_sequence = P.ReverseSequence(batch_dim = 1, seq_dim=0)
        self.concat = P.Concat(axis=2)
        self.squeeze = P.Squeeze(axis=0)
        self.rnn = P.DynamicGRUV2()
        self.text_len = max_length
        self.cast = P.Cast()
        

    def construct(self, x, seq_lens):
        '''
        BidirectionGRU construction

        Args:
            x(Tensor): BidirectionGRU input

        Returns:
            output(Tensor): rnn output
            hidden(Tensor): hidden state
        '''

        x = self.cast(x, mindspore.float16)
        y1, _, _, _, _, _ = self.rnn(x, self.weight_i, self.weight_h, self.bias_i, self.bias_h, None, self.init_h)
        # print("x", x.shape)
        bw_x = self.reverse_sequence(x, seq_lens)
        y1_bw, _, _, _, _, _ = self.rnn(bw_x, self.weight_bw_i,
                                        self.weight_bw_h, self.bias_bw_i, self.bias_bw_h, None, self.init_bw_h)
        y1_bw = self.reverse_sequence(y1_bw, seq_lens)
        output = self.concat((y1, y1_bw))
        hidden = self.concat((y1[self.text_len-1:self.text_len:1, ::, ::],
                              y1_bw[self.text_len-1:self.text_len:1, ::, ::]))
        #hidden = self.squeeze(hidden)                      
        return output, hidden

