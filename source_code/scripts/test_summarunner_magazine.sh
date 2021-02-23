logdir=logs/test/summarunner/
mkdir -p ${logdir}
exec > >(ts "%m/%d/%Y %H:%M:%S"| tee ${logdir}/magazine_test.log) 2>&1 
set -x

TEST_PATH='./datasets/kor_data/사설잡지/test.jsonl'
VOCAB_PATH='./datasets/kor_data/사설잡지/word_index_magazine.pkl'

TEST_BATCH_SIZE=64

CKPT_PATH='./checkpoint/summarunner/magazine/summarunnerepoch=15_val_loss=0.368_magazine.ckpt'
OUTPUT_PATH='./outputs/summarunner/magazine/'
RESULT_PATH='./result/summarunner/magazine'
TOPK=3

python src/extractive/summarunner/test.py \
    --test_path=${TEST_PATH}\
    --vocab_path=${VOCAB_PATH}\
    --test_batch_size=${TEST_BATCH_SIZE}\
    --ckpt_path=${CKPT_PATH}\
    --topk=${TOPK}\
    --output_path=${OUTPUT_PATH}

python src/extractive/summarunner/eval.py \
    --ref_path=${OUTPUT_PATH}/abs_ref \
    --hyp_path=${OUTPUT_PATH}/hyp \
    --result_path=${RESULT_PATH}