#!/bin/bash
# test dataset
python src/reader.py --task_name=match_kn_gene \
                     --max_seq_len=256 \
                     --vocab_path=data/char.dict \
                     --input_file=data/test.txt \
                     --output_file=../../test.mindrecord