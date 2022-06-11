import os
import pickle
from collections import Counter, defaultdict
from typing import List, Tuple

import gluonnlp as nlp
import torch
import numpy as np
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

## TODO: create cache file


class TokenizedDataset(Dataset):
    """ Dataset for Content Selection """

    def __init__(self, path: str, max_seqlen: int = 512, ignore_index=-100) -> None:
        super(TokenizedDataset, self).__init__()
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        self.max_len = max_seqlen

        _, self.vocab = get_pytorch_kobert_model()
        tok = get_tokenizer()
        self.tokenizer = nlp.data.BERTSPTokenizer(tok, self.vocab, lower=False)

        if "train" in path:
            self.data["token"] = self.data["token"][:100000]
            self.data["tgt"] = self.data["tgt"][:100000]

        self.tokens = self.data["token"]
        self.labels = self.data["tgt"]

        self.cls_idx = self.vocab["[CLS]"]
        self.pad_idx = self.vocab["[PAD]"]
        self.sep_idx = self.vocab["[SEP]"]
        self.mask_idx = self.vocab["[MASK]"]
        self.ignore_idx = ignore_index

    def add_special_token(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def add_special_token_label(self, labels):
        return [self.ignore_idx] + labels + [self.ignore_idx]

    def add_pad(self, token_ids: List[int]) -> List[int]:
        diff = self.max_len - len(token_ids)
        if diff > 0:
            token_ids += [self.pad_idx] * diff
        else:
            token_ids = token_ids[: self.max_len - 1] + [self.sep_idx]
        return token_ids

    def add_pad_label(self, labels):
        diff = self.max_len - len(labels)
        if diff > 0:
            labels += [self.pad_idx] * diff
        else:
            labels = labels[: self.max_len - 1] + [self.ignore_idx]
        return labels

    def idx2mask(self, token_ids: List[int]) -> List[bool]:
        return [token_id != self.pad_idx for token_id in token_ids]

    def label2mask(self, labels: List[int]) -> List[bool]:
        return [label != self.ignore_idx for label in labels]

    def __len__(self) -> int:
        return len(self.data["token"])

    def __getitem__(self, idx: int) -> Tuple[torch.tensor]:
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokens[idx])
        token_ids_padded = self.add_pad(self.add_special_token(token_ids))

        labels = self.labels[idx]
        labels_special = self.add_special_token_label(labels)
        labels_padded = self.add_pad_label(labels_special)

        att_mask_token = self.idx2mask(token_ids_padded)
        att_mask_label = self.label2mask(labels_padded)  # ignore_idx 부분도 masking
        att_mask = list(np.array(att_mask_token) & np.array(att_mask_label))

        assert len(token_ids_padded) == self.max_len
        assert len(att_mask) == self.max_len
        assert len(labels_padded) == self.max_len

        return {
            "input_ids": torch.tensor(token_ids_padded),
            "attention_mask": torch.tensor(att_mask),
            "labels": torch.tensor(labels_padded),
        }
