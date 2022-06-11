logdir=logs/test/textrank/
mkdir -p ${logdir}
exec > >(ts "%m/%d/%Y %H:%M:%S"| tee ${logdir}/magazine_test.log) 2>&1 
set -x

TEST_PATH='./datasets/kor_data/사설잡지/test.jsonl'

OUTPUT_PATH='./outputs/textrank/magazine/'
RESULT_PATH='./result/textrank/magazine'
TOPK=3

python ssrc/extractive/textrank/test.py \
    --test_path=${TEST_PATH}\
    --topk=${TOPK}\
    --output_path=${OUTPUT_PATH}
    
python src/extractive/textrank/eval.py \
    --ref_path=${OUTPUT_PATH}/abs_ref \
    --hyp_path=${OUTPUT_PATH}/hyp \
    --result_path=${RESULT_PATH}