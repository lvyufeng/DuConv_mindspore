#!/bin/bash
# test dataset
python src/reader.py --task_name=match_kn_gene \
                     --max_seq_len=256 \
                     --vocab_path=data/gene.dict \
                     --input_file=data/build.train.txt \
                     --output_file=../../train.mindrecord
python src/reader.py --task_name=match_kn_gene \
                     --max_seq_len=256 \
                     --vocab_path=data/gene.dict \
                     --input_file=data/build.dev.txt \
                     --output_file=../../dev.mindrecord
python src/reader.py --task_name=match_kn_gene \
                     --max_seq_len=256 \
                     --vocab_path=data/gene.dict \
                     --input_file=data/build.test.txt \
                     --output_file=../../test.mindrecord

mv ../../train.mindrecord* ./data
mv ../../dev.mindrecord* ./data
mv ../../test.mindrecord* ./data
