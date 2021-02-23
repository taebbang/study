set -x

BERT_DATA_PATH='./datasets/kor_data/bertabs_data_law/law'
OUTPUT_PATH='./outputs/bertsumabs/law/law'
CHECKPOINT_PATH='./checkpoint/bertsum_original/abs/law/model_step_35000.pt'
LOGDIR='./logs/bertsum/abs/law.log'

python src/bertsum/train.py \
    -task abs \
    -mode test \
    -batch_size 3000 \
    -test_batch_size 3000 \
    -bert_data_path ${BERT_DATA_PATH} \
    -log_file ${LOGDIR} \
    -sep_optim true \
    -use_interval true \
    -visible_gpus 0 \
    -max_pos 512 \
    -max_tgt_len 300 \
    -alpha 0.95 \
    -min_length 50 \
    -report_rouge false \
    -result_path ${OUTPUT_PATH} \
    -test_from ${CHECKPOINT_PATH}

exec > >(ts "%m/%d/%Y %H:%M:%S"| tee -a ${LOGDIR}) 2>&1 
python src/bertsum/cal_kor_rouge.py \
    --candidate_path './outputs/bertsumabs/law/law.35000.candidate' \
    --save_path './result/bertsumabs/law/'
