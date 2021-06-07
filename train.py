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
# from lr_schedule import CustomWarmUpLR,create_dynamic_lr
from lr_schedule import CustomWarmUpLR#dynamic_lr,

import data_provider as reader
from args import base_parser
from args import print_arguments
from mindspore import context
import mindspore.common.dtype as mstype
from retrieval_model import RetrievalModel
from config import cfg, net_cfg
from mindspore.nn import warmup_lr
from mindspore.profiler import Profiler

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
                                      do_lower_case=args.do_lower_case)#继承DataProcessor
    args.voc_size = len(open(args.vocab_path, 'r').readlines())
    if args.voc_size % 16 != 0:
        args.voc_size = (args.voc_size // 16) * 16 + 16

    num_labels = len(processor.get_labels())
    print("num_labels: ", num_labels)
    print("voc_size: ", args.voc_size)
    train_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='train',
        epoch=1,
        shuffle=True)
    num_train_examples = processor.get_num_examples(phase='train')
    is_training = args.is_training
    model = RetrievalModel(config = net_cfg, voc_size=args.voc_size, num_labels = num_labels, is_training = is_training) 
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    max_train_steps = args.epoch * num_train_examples // args.batch_size
    warmup_steps = int(max_train_steps * args.warmup_proportion)
    step_per_epoch = num_train_examples // args.batch_size
    # lr = warmup_lr(args.learning_rate, max_train_steps, step_per_epoch, 30)
    #print("warmup_steps:",warmup_steps)#269703PYNATIVE   GRAPH
    args.learning_rate = args.learning_rate * 0.5
    lr = Tensor(CustomWarmUpLR(d_model=256, learning_rate=args.learning_rate, warmup_steps=warmup_steps, training_steps=max_train_steps), mstype.float32)
    # lr = CustomWarmUpLR(d_model=256, learning_rate=args.learning_rate, warmup_steps=warmup_steps, max_train_steps=max_train_steps)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=5)
    # context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=5, save_graphs=True, save_graphs_path='../new_graph/')
    #lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                #   training_steps=num_train_examples*args.epoch,
                                #   learning_rate=cfg.lr_schedule.learning_rate,
                                #   warmup_steps=cfg.lr_schedule.warmup_steps,
                                #   hidden_size=net_cfg.hidden_size,
                                #   start_decay_step=cfg.lr_schedule.start_decay_step,
                                #   min_lr=cfg.lr_schedule.min_lr), mstype.float32)
    # lr = dynamic_lr(base_lr = 0.001, num_epochs=args.epoch, warmup_step=300, warmup_ratio = 1/3.0, base_step = num_train_examples // args.batch_size)
    
    params = model.trainable_params()
    # decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
    # other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
    # group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
    #                 {'params': other_params, 'weight_decay': 0.0}]
    # optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    # for i in params:
    #     print("name",i.name)
    # print("params", params)#args.learning_rate 1e-6
    optimizer = nn.Adam(params, learning_rate= lr, weight_decay =args.weight_decay)#GRAPH_MODE
    
    
    # profiler = Profiler()
    net_with_criterion = WithLossCell(model, criterion)
    train_network = nn.TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    time_begin = time.time()
    for epoch in range(args.epoch):
        for batch_id,data in enumerate(train_data_generator()):
            context_ids, context_pos_ids, context_segment_ids, context_attn_mask,kn_ids, labels, context_next_sent_index, seq_lengths = data
            # for i in kn_ids:
            #     print(i)
            #     print(np.array(i).shape)
            #     break
            # print("kn_ids:", np.array(kn_ids).shape)
            # print("seq_lengths:", np.array(seq_lengths).shape)#大小等于batch_size  batch_size*1
            
            context_ids_ = Tensor(np.squeeze(np.array(context_ids)),mindspore.int32)
            context_pos_ids_ = Tensor(np.squeeze(np.array(context_pos_ids)),mindspore.int32)
            context_segment_ids_ = Tensor(np.squeeze(np.array(context_segment_ids)),mindspore.int32)
            context_attn_mask_ = Tensor(np.squeeze(np.array(context_attn_mask)),mindspore.float16)
            labels_ = Tensor(np.squeeze(np.array(labels)),mindspore.int32)
            kn_ids_ = Tensor(np.squeeze(np.array(kn_ids)),mindspore.int32)

            context_next_sent_index_ = Tensor(np.squeeze(np.array(context_next_sent_index)),mindspore.int32)
            seq_lengths_ = Tensor(np.squeeze(np.array(seq_lengths)),mindspore.int32)
            # print("context_next_sent_index_", context_next_sent_index_)
            # if batch_id == 5:
            #     break
            # print("seq_lengths_:", seq_lengths_.shape)
            stack = P.Stack(axis=1)
            context_attn_mask_ = stack([context_attn_mask_]*8)
            #print("context_attn_mask_",context_attn_mask_.shape)
            #print("context_attn_mask_:", context_attn_mask_.shape)
            if context_ids_.shape[0]!= int(args.batch_size):
                break
            loss = train_network(kn_ids_, context_next_sent_index_,context_ids_, context_pos_ids_, context_segment_ids_, context_attn_mask_,seq_lengths_,labels_ )
            if batch_id % 100 == 0: 
                time_end = time.time()
                used_time = time_end - time_begin
                current_example, current_epoch = processor.get_train_progress()
                print('Epoch:', '%04d' % (epoch + 1), 'progress: ', current_example,'/', num_train_examples ,'Step:', '%04d' % (batch_id + 1), 'loss =', '{:.6f}'.format(loss.asnumpy()),"speed: %f steps/s",100 / used_time)
                time_begin = time.time()
            if batch_id % args.save_steps == 0: 
                print('Epoch:', '%04d' % (epoch + 1), 'Step:', '%04d' % (batch_id + 1), 'loss =', '{:.6f}'.format(loss.asnumpy()))
                temp = str(epoch + 1)+"_"+str(batch_id+1)
                model_path = os.path.join(args.checkpoints, str(temp))
                mindspore.save_checkpoint(net_with_criterion,model_path+'.ckpt')
                print('Save:',model_path+'.ckpt')  
    #         if batch_id == 5:  
    #             break
    #     break
    # profiler.analyse()

        
if __name__ == '__main__':
    args = base_parser()
    # print_arguments(args)
    main(args)