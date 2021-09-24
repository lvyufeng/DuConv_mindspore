#!/bin/bash

python train.py --epoch=30 \
                --task_name=match_kn_gene \
                --max_seq_length=256 \
                --batch_size=128 \
                --train_data_file_path=data/train.mindrecord \
                --save_checkpoint_path=save_model/ > train.log &