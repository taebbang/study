logdir=logs/test/pointergenerator/
logfile=law_test.log
mkdir -p ${logdir}
exec > >(ts "%m/%d/%Y %H:%M:%S"| tee ${logdir}/${logfile}) 2>&1 
set -x


TEST_PATH='./datasets/kor_data/법률문서/test.jsonl'
CHECKPOINT_PATH_COV="./checkpoint/pointergenerator/law/sp-covloss/epoch=72.ckpt"
VOCAB_PATH='./datasets/kor_data/법률문서/train_sp_50000.vocab'
ROUGE_PATH='./result/pointergenerator/law/'
OUTPUT_PATH='./outputs/pointergenerator/law/'
BATCH_SIZE=256

python src/abstractive/pointergenerator/inference.py \
    --test_dataset_path=${TEST_PATH}\
    --vocab_dir=${VOCAB_PATH}\
    --checkpoint_path=${CHECKPOINT_PATH_COV}\
    --output_path=${OUTPUT_PATH}\
    --rouge_path=${ROUGE_PATH}\
    --batch_size=${BATCH_SIZE}\
    --use_sentencepiece\
    --use_gpu\
    --ptr_gen
    