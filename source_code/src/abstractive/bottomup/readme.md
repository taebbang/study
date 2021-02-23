# Content Selection

## Data 설명
- ```/datasets/contentselection/select_train_tokenized.pickle``` : 1000개의 training data
- ```/datasets/contentselection/select_val_tokenized.pickle``` : 100개의 validation data
- BERTTokenizer로 토큰화된 dictionary에 ['label'] 추가된 형태
- ['input_ids']는 CNN-DailyMail의 토큰화된 데이터이다.
- ['labels']는 위 ['input_ids']에 대응하는 masking sequence이다.
  - ['input_ids']와 같은 길이를 가지며, 0 또는 1로 구성되어 있다.
  - 0은 masking된 토큰이고, 1은 select된 토큰이다.
- type: torch.tensor

### 예시
```python
# Dataset Class에서 squeeze 진행함

data = {
    'input_ids' : tensor([101, 1006, ...]), # (1, 512)
    'token_type_ids' : tensor([0, 0, ...]), # (1, 512)
    'attention_mask' : tensor([1, 1, ...]), # (1, 512)
    'labels' : tensor([1, 0, 0, ...])       # (512)
}
```


<!-- ## TODO
- [x] label 생성 코드 짜기(preprocessing.py)
- [x] 기존 Dataset 대신 원본 src, tgt으로 content selection 진행 (UNK토큰때문)
- [x] Content Selector 학습을 위한 데이터를 따로 pickle로 저장하기
- [x] BERT Classifier 붙이기 -->
