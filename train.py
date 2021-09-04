import mindspore
import numpy as np
from mindspore import Tensor, context
from src.model import Retrieval
from src.bert import BertConfig
    
if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE)
    config = BertConfig()
    net = Retrieval(config, True)
    input_ids = Tensor(np.random.uniform(0, 512, (1, 256)).astype(np.int32))
    segment_ids = Tensor(np.ones((1, 256)).astype(np.int32))
    kn_ids = Tensor(np.random.uniform(0, 512, (1, 256)).astype(np.int32))
    seq_length = Tensor([256], mindspore.int32)

    logits = net(input_ids, segment_ids, kn_ids, seq_length)