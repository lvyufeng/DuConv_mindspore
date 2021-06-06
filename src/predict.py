# -*- coding: utf-8 -*-
"""
File: train.py
"""

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
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
import data_provider as reader
from args import base_parser
from args import print_arguments
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from retrieval_model import RetrievalModel
from config import cfg, net_cfg

class WithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
    def construct(self, *args):
        outputs = self._backbone(*args[:-1])
        return self._loss_fn(outputs, args[-1])

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


def main(args):

    VOC_DICT = load_dict(args.vocab_path)
    processor = reader.MatchProcessor(data_dir=args.data_dir,
                                      task_name=args.task_name,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case)
    args.voc_size = len(open(args.vocab_path, 'r').readlines())
    if args.voc_size % 16 != 0:
        args.voc_size = (args.voc_size // 16) * 16 + 16
    num_labels = len(processor.get_labels())
    print("num_labels: ", num_labels)
    infer_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='test',
        epoch=1,
        shuffle=False)
    num_test_examples = processor.get_num_examples(phase='test')

    is_training = args.is_training
    model = RetrievalModel(config = net_cfg, voc_size=args.voc_size, num_labels = num_labels, is_training = is_training) 
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=5)
    # context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=7)
    param_dict = load_checkpoint(args.init_checkpoint)
    load_param_into_net(model, param_dict)
    out_scores = open(args.output, 'w')
    num = 0
 
    for batch_id,data in enumerate(infer_data_generator()):
        context_ids, context_pos_ids, context_segment_ids, context_attn_mask,kn_ids, labels, context_next_sent_index, seq_lengths = data
        #print("context_ids:", context_ids.shape)
        context_ids_ = Tensor(np.squeeze(np.array(context_ids)),mindspore.int32)
        context_pos_ids_ = Tensor(np.squeeze(np.array(context_pos_ids)),mindspore.int32)
        context_segment_ids_ = Tensor(np.squeeze(np.array(context_segment_ids)),mindspore.int32)
        context_attn_mask_ = Tensor(np.squeeze(np.array(context_attn_mask)),mindspore.float16)
        labels_ = Tensor(np.squeeze(np.array(labels)),mindspore.int32)
        kn_ids_ = Tensor(np.squeeze(np.array(kn_ids)),mindspore.int32)
        context_next_sent_index_ = Tensor(np.squeeze(np.array(context_next_sent_index)),mindspore.int32)
        seq_lengths_ = Tensor(np.squeeze(np.array(seq_lengths)),mindspore.int32)
        # print("context_attn_mask_:", context_attn_mask_.shape)
        stack = P.Stack(axis=1)
        context_attn_mask_ = stack([context_attn_mask_]*8)
        # print("context_attn_mask_",context_attn_mask_.shape)
        #print("context_attn_mask_:", context_attn_mask_.shape)
        if context_ids_.shape[0]!= int(args.batch_size):
            break
        predict = model(kn_ids_, context_next_sent_index_, context_ids_, context_pos_ids_, context_segment_ids_, context_attn_mask_, seq_lengths_)
        softmax = nn.Softmax()
        output = softmax(predict)
        for elem in output:
            out_scores.write(str(elem[1]) + '\n')
    


        
if __name__ == '__main__':
    args = base_parser()
    # print_arguments(args)
    main(args)