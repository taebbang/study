TRAIN_PATH='./datasets/kor_data/total_data_1018/train.jsonl'
VAL_PATH='./datasets/kor_data/total_data_1018/dev.jsonl'
TEST_PATH='./datasets/kor_data/total_data_1018/test.jsonl'
CHECKPOINT_PATH_NO_COV="./checkpoint/pointergenerator/kor_data_1018/mecab/no_covloss/"
LOG_DIR="./logs/pointergenerator/kor_data_1018/mecab/no_covloss/"

BATCH_SIZE=55
VALIDATION_STEP=400 # Float: epoch, Int: step
STEPS=30000

# Do not use coverage loss
python src/abstractive/pointergenerator/train.py \
    --train_dataset=${TRAIN_PATH}\
    --val_dataset=${VAL_PATH}\
    --test_dataset=${TEST_PATH}\
    --train_batch_size=${BATCH_SIZE}\
    --ptr_gen\
    --checkpoint_dir=${CHECKPOINT_PATH_NO_COV}\
    --log_dir=${LOG_DIR}\
    --val_interval=${VALIDATION_STEP}\
    --steps=${STEPS}

CHECKPOINT_PATH_COV="./checkpoint/pointergenerator/kor_data_1018/mecab/covloss/"

# Use coverage loss
STEPS=40000
VALIDATION_STEP=200
python src/abstractive/pointergenerator/train.py \
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
    
