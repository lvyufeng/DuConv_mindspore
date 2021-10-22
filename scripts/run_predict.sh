#!/bin/bash
if [ $# -ne 1 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "sh build_dataset.sh [TASK_NAME]"
    echo "for example: sh scripts/build_dataset.sh match_kn_gene"
    echo "TASK_TYPE including [match, match_kn, match_kn_gene]"
    echo "=============================================================================================================="
exit 1
fi
TASK_NAME=$1

candidate_file=data/candidate.test.txt
predict_path=output
load_checkpoint_path=save_model
for file in `ls -t ${load_checkpoint_path}`
do
    load_checkpoint_file=${load_checkpoint_path}/${file}
    score_file=${predict_path}/score.${file}.txt
    result_file=${predict_path}/result.${file}.txt

    python predict.py --task_name=${TASK_NAME} \
                    --max_seq_length=128 \
                    --batch_size=100 \
                    --eval_data_file_path=data/test.mindrecord \
                    --load_checkpoint_path=${load_checkpoint_file} \
                    --save_file_path=${score_file}

    python src/utils/extract.py ${candidate_file} ${score_file} ${result_file}

    # step 6: if the original file has answers, you can run the following command to get result
    # if the original file not has answers, you can upload the ./output/test.result.final 
    # to the website(https://ai.baidu.com/broad/submission?dataset=duconv) to get the official automatic evaluation
    python src/eval.py ${result_file} > ${predict_path}/predict.${file}.log
done
