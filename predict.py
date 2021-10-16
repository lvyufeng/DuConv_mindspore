import os
import time
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
import mindspore.ops.operations as P
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model import RetrievalWithSoftmax
from src.bert import BertConfig
from src.dataset import create_dataset

def parse_args():
    """set and check parameters"""
    parser = argparse.ArgumentParser(description='train duconv')
    parser.add_argument('--task_name', type=str, default='match_kn', choices=['match', 'match_kn', 'match_kn_gene'],
                        help='task name for training')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Number of word of the longest seqence. (default: %(default)d)')
    parser.add_argument('--batch_size', type=int, default=8096,
                        help='Total token number in batch for training. (default: %(default)d)')
    parser.add_argument('--vocab_size', type=int, default=14373,
                        help='Total token number in batch for training. (default: %(default)d)')
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--load_checkpoint_path", type=str, default="", help="Save checkpoint path")
    parser.add_argument("--save_file_path", type=str, default="", help="Save checkpoint path")
    args = parser.parse_args()

    return args

def run_duconv():
    """run duconv task"""
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, max_call_depth=10000)
    use_kn = True if "kn" in args.task_name else False
    config = BertConfig(seq_length=args.max_seq_length, vocab_size=args.vocab_size)
    dataset = create_dataset(args.batch_size, data_file_path=args.eval_data_file_path,
                             do_shuffle=False, use_knowledge=use_kn)
    steps_per_epoch = dataset.get_dataset_size()
    print(steps_per_epoch)

    network = RetrievalWithSoftmax(config, use_kn)
    param_dict = load_checkpoint(args.load_checkpoint_path)
    not_loaded = load_param_into_net(network, param_dict)
    print(not_loaded)
    network.set_train(False)

    f = open(args.save_file_path, 'w')
    iterator = dataset.create_tuple_iterator()
    for item in iterator:
        output = network(*item[:-1])
        for i in output:
            f.write(str(i[1]) + '\n')
            f.flush()
    f.close()

if __name__ == '__main__':
    run_duconv()