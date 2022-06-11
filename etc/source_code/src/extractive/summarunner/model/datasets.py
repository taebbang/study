import json
import numpy as np
from collections import Counter, defaultdict
from konlpy.tag import Mecab
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from .types_ import *


class SumDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            jsonl = list(f)

        self.data = []
        for json_str in jsonl:
            self.data.append(json.loads(json_str))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc = self.data[idx]["article_original"]
        ext_indices = self.data[idx]["extractive"]
        summaries = self.data[idx]["abstractive"]
        doc_id = self.data[idx]["id"]

        return doc, ext_indices, summaries, doc_id


class Feature:
    def __init__(self, word_index, tokenizer):
        self.word_index = word_index
        self.index_word = {idx: word for word, idx in word_index.items()}
        assert len(self.word_index) == len(self.index_word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.word_index)

    def index_to_word(self, idx):
        return self.index_word[idx]

    def word_to_index(self, w):
        if w in self.word_index:
            return self.word_index[w]
        else:
            return self.UNK_IDX

    ###################
    # Create Features #
    ###################
    def make_features(
        self,
        docs,
        ext_idx_list,
        summaries_list,
        doc_trunc=50,
        sent_trunc=128,
        split_token="\n",
    ):

        # trunc document
        # 문서 내 doc_trunc 문장 개수까지 가져옴
        sents_list, targets, doc_lens, ext_sums, abs_sums = [], [], [], [], []
        for doc, ext_indices, abs_sum in zip(docs, ext_idx_list, summaries_list):
            labels = []
            for idx in range(len(doc)):
                if idx in ext_indices:
                    labels.append(1)
                else:
                    labels.append(0)

            max_sent_num = min(doc_trunc, len(doc))
            sents = doc[:max_sent_num]
            labels = labels[:max_sent_num]
            ext_sum = [sent for sent, label in zip(sents, labels) if label == 1]

            sents_list.extend(sents)
            targets.extend(labels)
            doc_lens.append(len(sents))
            ext_sums.append(ext_sum)
            abs_sums.append(abs_sum)

        # trunc or pad sent
        # 문장 내 sent_trunc 단어 개수까지 가져옴
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = self.tokenizer.morphs(sent)
            # words = [word for word in words if len(word) > 1]
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.PAD_IDX for _ in range(max_sent_len - len(sent))] + [
                self.word_to_index(w) for w in sent
            ]
            features.append(feature)

        return features, targets, doc_lens, ext_sums, abs_sums, docs

    def make_predict_features(
        self,
        docs,
        sent_trunc=128,
        doc_trunc=50,
        split_token=". ",
    ):

        sents_list, doc_lens = [], []
        for doc in docs:
            sents = doc.split(split_token)
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            sents_list.extend(sents)
            doc_lens.append(len(sents))

        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = self.tokenizer.morphs(sent)
            # words = [word for word in words if len(word) > 1]
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.PAD_IDX for _ in range(max_sent_len - len(sent))] + [
                self.word_to_index(w) for w in sent
            ]
            features.append(feature)

        return features, doc_lens


def build_vocab(dataset: JSONType, stopwords: Optional[List[str]] = None, num_words: int = 40000):
    # 0. tokenizer
    tokenizer = Mecab()

    # 1. tokenization
    all_tokens = []
    for data in tqdm(dataset):
        sents = data["article_original"]
        for sent in sents:
            tokens = tokenizer.morphs(sent)
            if stopwords:
                all_tokens.extend([token for token in tokens if token not in stopwords])
            else:
                all_tokens.extend(tokens)

    # 2. build vocab
    vocab = Counter(all_tokens)
    vocab = vocab.most_common(num_words)

    # 3. add pad & unk tokens
    word_index = defaultdict()
    word_index["<PAD>"] = 0
    word_index["<UNK>"] = 1

    for idx, (word, _) in enumerate(vocab, 2):
        word_index[word] = idx

    index_word = {idx: word for word, idx in word_index.items()}

    return word_index, index_word


def collate_fn(batch, feature):
    docs = [entry[0] for entry in batch]
    labels_list = [entry[1] for entry in batch]
    summaries_list = [entry[2] for entry in batch]
    doc_ids = [entry[3] for entry in batch]

    features, targets, doc_lens, ext_sums, abs_sums, orgin_docs = feature.make_features(
        docs, labels_list, summaries_list
    )

    docs = []
    labels = []
    start = 0
    pad_dim = len(features[0])
    max_doc_len = max(doc_lens)
    for doc_len in doc_lens:
        stop = start + doc_len
        doc = features[start:stop]
        target = targets[start:stop]
        start = stop

        doc = torch.LongTensor(doc)
        if len(doc) == max_doc_len:
            docs.append(doc.unsqueeze(0))
        else:
            pad = torch.zeros(max_doc_len - doc_len, pad_dim, dtype=torch.long)
            docs.append(torch.cat([doc, pad]).unsqueeze(0))

        if len(target) == max_doc_len:
            labels.append(torch.FloatTensor(target).unsqueeze(0))
        else:
            pad = torch.zeros(max_doc_len - doc_len)
            target = torch.FloatTensor(target)
            labels.append(torch.cat([target, pad]).unsqueeze(0))

    docs = torch.cat(docs, dim=0)
    labels = torch.cat(labels, dim=0)
    targets = torch.FloatTensor(targets)
    doc_lens = torch.LongTensor(doc_lens)
    return docs, labels, doc_lens, max_doc_len, ext_sums, abs_sums, orgin_docs, doc_ids