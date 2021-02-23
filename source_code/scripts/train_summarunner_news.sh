TRAIN_PATH='./datasets/kor_data/신문기사/train.jsonl'
DEV_PATH='./datasets/kor_data/신문기사/dev.jsonl'
TEST_PATH='./datasets/kor_data/신문기사/test.jsonl'
#VOCAB_PATH='./src/extractive/summarunner/utils/word_index_v02.pkl'

TRAIN_BATCH_SIZE=256
DEV_BATCH_SIZE=256
TEST_BATCH_SIZE=256

CHECK_POINT_DIR='./checkpoint/summarunner/news'
LOG_DIR='./logs/summarunner/'

python src/extractive/summarunner/train.py \
    --train_path=${TRAIN_PATH}\
    --dev_path=${DEV_PATH}\
    --test_path=${TEST_PATH}\
    --train_batch_size=${TRAIN_BATCH_SIZE}\
    --eval_batch_size=${DEV_BATCH_SIZE}\
    --test_batch_size=${TEST_BATCH_SIZE}\
    --checkpoint_dir=${CHECK_POINT_DIR}\
    --log_dir=${LOG_DIR}