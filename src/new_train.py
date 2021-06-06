import os
import time
import numpy as np
import multiprocessing
import mindspore
import mindspore.nn as nn
from mindspore.nn.optim import Adam
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.callback import Callback, TimeMonitor
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
import mindspore.ops.operations as P
import new_data_provider as reader
from args import base_parser
from args import print_arguments
from mindspore import context
def load_dict(vocab_dict): 
    VOC_DICT = {}
    """
    load vocabulary dict
    """
    idx = 0
    for line in open(vocab_dict): 
        line = line.strip()
        VOC_DICT[line] = idx
        idx += 1
    return VOC_DICT
VOC_DICT = load_dict("../dict/gene.dict")
processor = reader.MatchProcessor(data_dir="../data",
                                  task_name="match_kn_gene",
                                  vocab_path="../dict/gene.dict",
                                  max_seq_len=256,
                                  do_lower_case=True)
voc_size = len(open("../dict/gene.dict", 'r').readlines())
num_labels = len(processor.get_labels())
print("num_labels: ", num_labels)
print("voc_size: ", voc_size)
train_data_generator = processor.data_generator(
    batch_size=0,
    phase='train',
    epoch=1,
    shuffle=True)
num_train_examples = processor.get_num_examples(phase='train')

import mindspore.dataset as ds
from mindspore.dataset import GeneratorDataset

my_generator = ds.GeneratorDataset(train_data_generator,["context_ids", "context_pos_ids", "context_segment_ids", "context_attn_mask", "kn_ids", "labels", "context_next_sent_index"])

iterator = my_generator.create_dict_iterator()
for item in iterator:
    context_ids, context_pos_ids, context_segment_ids, context_attn_mask,kn_ids, labels, context_next_sent_index = item
#     context_ids_ = Tensor(np.squeeze(np.array(context_ids)),mindspore.int32)
    print(context_ids.shape)
    break