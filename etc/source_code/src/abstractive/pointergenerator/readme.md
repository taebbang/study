# Pointer Generator Network

Pytorch lightning implementation of [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

## Results

|                                              | ROUGE-1(F1) | ROUGE-2(F1) | ROUGE-L(F1) |
| :------------------------------------------: | :---------: | :---------: | :---------: |
| **Model** vs **Abstractive** <br />summaries |    0.378    |    0.142    |    0.396    |

## TODO

- [x] implement seq2seq + attention module
- [x] connect dataset with model
- [x] add pointer-generator
- [x] add coverage mechanism
- [x] add decoding mechanism
- [x] add rouge score
