#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=3
export GLOG_v=3
#  /usr/local/sbin/npu_smi 
# task_name can select from ["match", "match_kn", "match_kn_gene"]
# match task: do not use knowledge info (goal and knowledge) for retrieval model
# match_kn task: use knowledge info (goal and knowledge) for retrieval model
# match_kn_gene task: 1) use knowledge info (goal and knowledge) for retrieval model;
#                     2) generalizes target_a/target_b of goal, replaces them with slot mark
# more information about generalization in match_kn_gene,
# you can refer to ./tools/convert_conversation_corpus_to_model_text.py
TASK_NAME="match_kn_gene"

if [ "$TASK_NAME" = "match_kn_gene" ]
then
  DICT_NAME="../dict/gene.dict"
  USE_KNOWLEDGE=1
  TOPIC_GENERALIZATION=1
  OUTPUT_PATH="../models/graph_test"
else
  echo "task name error, should be match_kn_gene"
fi

# in train stage, FOR_PREDICT=0
FOR_PREDICT=0

# put all data set that used and generated for training under this folder: INPUT_PATH
# for more details, please refer to the following data processing instructions
INPUT_PATH="../data"

# put the model file that saved in each stage under this folder: OUTPUT_PATH
# OUTPUT_PATH="./models"

# set python path according to your actual environment
# PYTHON_PATH="/home/ma-user/miniconda3/envs/Mindspore-1.1.1-python3.7-aarch64/bin/python3"
PYTHON_PATH="python"
# in train stage, use "train.txt" to train model, and use "dev.txt" to eval model
# the "train.txt" and "dev.txt" are the original data of DuConv and
# need to be placed in this folder: INPUT_PATH/resource/
# the following preprocessing will generate the actual data needed for model training
# DATA_TYPE = "train" or "dev"
DATA_TYPE=("train" "dev")

# candidate set
candidate_set_file=${INPUT_PATH}/candidate_set.txt
# echo ${DATA_TYPE[$0]}
# data preprocessing
# for ((i=0; i<${#DATA_TYPE[*]}; i++))
# do
#     corpus_file=${INPUT_PATH}/resource/${DATA_TYPE[$i]}.txt
#     sample_file=${INPUT_PATH}/resource/sample.${DATA_TYPE[$i]}.txt
#     candidate_file=${INPUT_PATH}/resource/candidate.${DATA_TYPE[$i]}.txt
#     text_file=${INPUT_PATH}/${DATA_TYPE[$i]}.txt
#     # step 1: build candidate set from session data for negative training cases and predicting candidates
#     if [ "${DATA_TYPE[$i]}"x = "train"x ]; then
#         ${PYTHON_PATH} ../src/build_candidate_set_from_corpus.py ${corpus_file} ${candidate_set_file}
#     fi
#     # fi 结束if循环
#     # step 2: firstly have to convert session data to sample data
#     ${PYTHON_PATH} ../src/convert_session_to_sample.py ${corpus_file} ${sample_file}

#     # # step 3: construct candidate for sample data
#     ${PYTHON_PATH} ../src/construct_candidate.py ${sample_file} ${candidate_set_file} ${candidate_file} 9

#     # # step 4: convert sample data with candidates to text data required by the model
#     ${PYTHON_PATH} ../src/convert_conversation_corpus_to_model_text.py ${candidate_file} ${text_file} ${USE_KNOWLEDGE} ${TOPIC_GENERALIZATION} ${FOR_PREDICT}

#     # # step 5: build dict from the training data, here we build character dict for model
#     if [ "${DATA_TYPE[$i]}"x = "train"x ]; then
#         ${PYTHON_PATH} ../src/build_dict.py ${text_file} ${DICT_NAME}
#     fi

# done
# data preprocessing
# --data_dir ${INPUT_PATH}/resource \
# step 5: train model, you can find the model file in OUTPUT_PATH after training
$PYTHON_PATH -u ../src/test.py --task_name ${TASK_NAME} \
                   --use_cuda \
                   --batch_size 64 \
                   --data_dir ${INPUT_PATH} \
                   --vocab_path ${DICT_NAME} \
                   --checkpoints ${OUTPUT_PATH} \
                   --save_steps 1000 \
                   --save_checkpoint_num 30 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.1 \
                   --validation_steps 1000000 \
                   --skip_steps 100 \
                   --learning_rate 1e-6 \
                   --epoch 30 \
                   --max_seq_len 256 \
                   --is_training True
