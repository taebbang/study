import copy
from typing import List, Tuple
import torch


class Collator:
    def __init__(
        self, dset: torch.utils.data.Dataset, ptr_gen: bool, max_len: int, use_sentencepiece: bool
    ) -> None:
        self.dataset = dset
        self.do_ptr_gen = ptr_gen
        self.max_len = max_len
        self.use_sentencepiece = use_sentencepiece

    def pad(
        self, sample: List[int], max_len: int, pad_idx: int, preserve_last: bool = False
    ) -> List[int]:
        diff = max_len - len(sample)
        if diff > 0:
            return sample + [pad_idx] * diff
        else:
            if preserve_last:
                # Preserve last token([STOP]) of target output
                return sample[: max_len - 1] + [sample[-1]]
            return sample[:max_len]

    def __call__(self, batch: Tuple[list]) -> Tuple[torch.LongTensor]:
        src, tgt, src_oov_vocabs, tgt_oov_vocabs = zip(*batch)

        src_len = torch.tensor([self.max_len if len(s) > self.max_len else len(s) for s in src])
        tgt_len = torch.tensor([self.max_len if len(s) > self.max_len else len(s) for s in tgt])

        src_max_len, tgt_max_len = max(src_len).item(), max(tgt_len).item()

        # 매 배치마다 vocab_with_oov 생성 -> 이걸 통해 oov 단어도 generation 가능
        if self.do_ptr_gen and not self.use_sentencepiece:
            oov_list = list(set(x for sublist in src_oov_vocabs for x in sublist.keys()))
            vocab_with_oov = copy.deepcopy(self.dataset.vocab)
            for i in range(len(oov_list)):
                vocab_with_oov[oov_list[i]] = len(self.dataset.vocab) + i

            src_oov, tgt_oov = [copy.deepcopy(i) for i in (src, tgt)]

            # 3중 반복문이지만 문장마다 oov가 많지 않기 때문에 실제 계산량은 별로 없을듯..? -> 체크해보기
            for i, oovs in enumerate(src_oov_vocabs):  # n: batch size
                for oov in oovs.keys():  # n: the number of OOV words in article
                    for idx in oovs[oov]:  # n : the number of appearance of the OOV word
                        src_oov[i][idx] = vocab_with_oov[oov]
                        # 결국 batch size 안의 모든 OOV word에 대해 반복문을 돌린 것과 동일(O(N))

            for i, oovs in enumerate(tgt_oov_vocabs):
                for oov in oovs.keys():
                    for idx in oovs[oov]:
                        tgt_oov[i][idx] = vocab_with_oov[oov]

        pad_idx = self.dataset.vocab["[PAD]"]

        src = torch.LongTensor([self.pad(s, src_max_len, pad_idx) for s in src])
        tgt_input = torch.LongTensor([self.pad(t[:-1], tgt_max_len, pad_idx) for t in tgt])

        src_padding_mask = src != pad_idx

        if self.do_ptr_gen and not self.use_sentencepiece:
            src_oov = torch.LongTensor([self.pad(s, src_max_len, pad_idx) for s in src_oov])
            tgt_output_oov = torch.LongTensor(
                [self.pad(t[1:], tgt_max_len, pad_idx, True) for t in tgt_oov]
            )
            tgt_padding_mask = tgt_output_oov != pad_idx
            return (
                src,
                tgt_input,
                tgt_output_oov,
                src_len,
                tgt_len,
                src_padding_mask,
                tgt_padding_mask,
                src_oov,
                vocab_with_oov,
            )
        else:
            tgt_output = torch.LongTensor(
                [self.pad(t[1:], tgt_max_len, pad_idx, True) for t in tgt]
            )
            tgt_padding_mask = tgt_output != pad_idx
            return (
                src,
                tgt_input,
                tgt_output,
                src_len,
                tgt_len,
                src_padding_mask,
                tgt_padding_mask,
                src,
                None,
            )

