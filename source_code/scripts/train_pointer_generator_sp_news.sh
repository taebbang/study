TRAIN_PATH='./datasets/kor_data/신문기사/train.jsonl'
VAL_PATH='./datasets/kor_data/신문기사/dev.jsonl'
TEST_PATH='./datasets/kor_data/신문기사/test.jsonl'
CHECKPOINT_PATH_NO_COV="./checkpoint/pointergenerator/news/sp-no-covloss/"
LOG_DIR="./logs/pointergenerator/sp-no-covloss/"

BATCH_SIZE=50
EVAL_BATCH_SIZE=850
VALIDATION_STEP=400 
STEPS=300000

# Do not use coverage loss
python src/abstractive/pointergenerator/train.py \
    --use_sentencepiece\
    --train_dataset=${TRAIN_PATH}\
    --val_dataset=${VAL_PATH}\
    --test_dataset=${TEST_PATH}\
    --train_batch_size=${BATCH_SIZE}\
    --ptr_gen\
    --checkpoint_dir=${CHECKPOINT_PATH_NO_COV}\
    --log_dir=${LOG_DIR}\
    --val_interval=${VALIDATION_STEP}\
    --steps=${STEPS}\
    --eval_batch_size=${EVAL_BATCH_SIZE}\
    --distributed_backend=ddp\
    --precision=32

CHECKPOINT_PATH_COV="./checkpoint/pointergenerator/news/sp-covloss/"

# Use coverage loss
STEPS=330000
VALIDATION_STEP=200
BATCH_SIZE=45
EVAL_BATCH_SIZE=800
echo use checkpoint `ls -tr ${CHECKPOINT_PATH_NO_COV} | tail -1`
python src/abstractive/pointergenerator/train.py \
    --use_sentencepiece\
    --train_dataset=${TRAIN_PATH}\
    --val_dataset=${VAL_PATH}\
    --test_dataset=${TEST_PATH}\
    --train_batch_size=${BATCH_SIZE}\
    --load_checkpoint_from=${CHECKPOINT_PATH_NO_COV}`ls -tr ${CHECKPOINT_PATH_NO_COV} | tail -1`\
    --checkpoint_dir=${CHECKPOINT_PATH_COV}\
    --log_dir=${LOG_DIR}\
    --ptr_gen\
    --coverage\
    --val_interval=${VALIDATION_STEP}\
    --steps=${STEPS}\
    --eval_batch_size=${EVAL_BATCH_SIZE}\
    --distributed_backend=ddp\
    --precision=32
    
