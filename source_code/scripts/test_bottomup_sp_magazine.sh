logdir=logs/test/bottomup/
logfile=magazine_test.log
mkdir -p ${logdir}
exec > >(ts "%m/%d/%Y %H:%M:%S"| tee ${logdir}/${logfile}) 2>&1 
set -x

TEST_PATH='./datasets/kor_data/사설잡지/test.jsonl'
CHECKPOINT_PATH_COV="./checkpoint/pointergenerator/magazine/sp-covloss/epoch=76.ckpt"
VOCAB_PATH='./datasets/kor_data/사설잡지/train_sp_50000.vocab'
ROUGE_PATH='./result/bottomup/magazine/'
OUTPUT_PATH='./outputs/bottomup/magazine/'
BATCH_SIZE=256

# bottomup parameter
CONTENT_SELECTION_PATH='./datasets/kor_data/contentselection_magazine/contentselection_test_cs.pickle'
ALPHA=0
BETA=0
NGRAM=0
CONTENTS_THRESHOLD=0.2

python src/abstractive/pointergenerator/inference.py \
    --test_dataset_path=${TEST_PATH}\
    --vocab_dir=${VOCAB_PATH}\
    --checkpoint_path=${CHECKPOINT_PATH_COV}\
    --rouge_path=${ROUGE_PATH}\
    --output_path=${OUTPUT_PATH}\
    --batch_size=${BATCH_SIZE}\
    --use_sentencepiece\
    --use_gpu\
    --ptr_gen\
    --content_selection_path=${CONTENT_SELECTION_PATH}\
    --length_penalty=${ALPHA}\
    --copy_penalty_beta=${BETA}\
    --no_repeat_ngram_size=${NGRAM}\
    --contents_threshold=${CONTENTS_THRESHOLD}
    