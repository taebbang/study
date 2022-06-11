import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter, defaultdict
from typing import List, Tuple
import json
import pickle
import os

PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[UNK]"
START_DECODING = "[START]"
STOP_DECODING = "[STOP]"

special_tokens = [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]


class TextDataset(Dataset):
    def __init__(
        self,
        path: str,
        max_vocab: int = 50000,
        mode: str = "train",
        vocab_dir: str = None,
        use_sentencepiece: bool = False,
    ):
        """
        데이터 불러오고 단어 만들기
        """
        self.use_sentencepiece = use_sentencepiece
        if self.use_sentencepiece:
            # Use sentencepiece tokenizer from SKT Bert(https://github.com/SKTBrain/KoBERT)
            import gluonnlp as nlp
            from kobert.utils import get_tokenizer
            from kobert.pytorch_kobert import get_pytorch_kobert_model
            from gluonnlp.data import SentencepieceTokenizer

            bertmodel, vocab = get_pytorch_kobert_model()
            tokenizer = get_tokenizer()
            self.tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        else:
            # Use Mecab tokenizer
            from konlpy.tag import Mecab

            self.tokenizer = Mecab()
        self.max_vocab = max_vocab

        folder_dir, file_dir = os.path.split(path)
        tok = "sp_" if use_sentencepiece else "mecab_"
        cached_dir = os.path.join(folder_dir, "cached_" + tok + file_dir)
        vocab_file_dir = os.path.splitext(path)[0] + "_" + tok + str(max_vocab) + ".vocab"

        if os.path.isfile(cached_dir):
            # 전처리된 파일이 있다면 불러오기
            with open(cached_dir, "rb") as f:
                (self.src, self.tgt, self.src_oov_vocabs, self.tgt_oov_vocabs,) = pickle.load(f)
            # 기존에 생성되었던 vocab file load
            if mode == "train":
                if not os.path.isfile(vocab_file_dir):
                    raise ValueError("Vocabulary file does not exist")
                with open(vocab_file_dir, "rb") as f:
                    self.vocab = pickle.load(f)
            elif mode in ["eval", "test"]:
                if vocab_dir is None:
                    dirs, vocab_file_dir = os.path.split(vocab_file_dir)
                    vocab_file_dir = vocab_file_dir.split("_")
                    vocab_file_dir[0] = "train"
                    vocab_file_dir = os.path.join(dirs, "_".join(vocab_file_dir))
                    if os.path.isfile(vocab_file_dir):
                        with open(vocab_file_dir, "rb") as f:
                            self.vocab = pickle.load(f)
                    else:
                        raise ValueError("Directory of Vocab file should be given")
                else:
                    with open(vocab_dir, "rb") as f:
                        self.vocab = pickle.load(f)
            else:
                raise ValueError(
                    "Invalid value given in 'mode' argument. Value should be in 'train', 'eval', or 'test'"
                )
        else:
            # 전처리된 파일이 없다면 새로 만들기
            print("Loading original text files ... ")
            with open(path, "r", encoding="utf-8") as f:
                jsonl = list(f)
            data = []
            for json_str in jsonl:
                data.append(json.loads(json_str))
            src = [d["article_original"] for d in data]
            src = [" ".join(s) for s in src]
            tgt = [d["abstractive"] for d in data]
            assert len(src) == len(tgt)

            # 문장 토크나이징하기
            if self.use_sentencepiece:
                src = [self.tokenizer.convert_tokens_to_ids(self.tokenizer(d)) for d in src]
                tgt = [self.tokenizer.convert_tokens_to_ids(self.tokenizer(d)) for d in tgt]
            else:
                src = [["_".join(t) for t in self.tokenizer.pos(d)] for d in src]
                tgt = [["_".join(t) for t in self.tokenizer.pos(d)] for d in tgt]

            if mode == "train":
                if vocab_dir is not None:
                    raise FileNotFoundError(
                        "argument vocab_dir should be used in evaluation or test mode"
                    )
                else:
                    # vocab 파일 생성
                    print("Making vocabulary file ... ")
                    self.vocab = self.get_vocab(src)
                    # input 파일이 위치한 폴더에 저장
                    with open(vocab_file_dir, "wb") as f:
                        pickle.dump(self.vocab, f)
                        print('Save Vocabulary file on "' + vocab_file_dir + '"')
            elif mode in ["eval", "test"]:
                if vocab_dir is None:
                    dirs, vocab_file_dir = os.path.split(vocab_file_dir)
                    vocab_file_dir = vocab_file_dir.split("_")
                    vocab_file_dir[0] = "train"
                    vocab_file_dir = os.path.join(dirs, "_".join(vocab_file_dir))
                    if os.path.isfile(vocab_file_dir):
                        with open(vocab_file_dir, "rb") as f:
                            self.vocab = pickle.load(f)
                    else:
                        raise ValueError(
                            "Directory of Vocab file should be given in evaluation or test mode"
                        )
                else:
                    with open(vocab_dir, "rb") as f:
                        self.vocab = pickle.load(f)
            else:
                raise ValueError(
                    "Invalid value given in 'mode' argument. Value should be in 'train', 'eval', or 'test'"
                )
            # 문장 전처리하기 (words -> indices)
            print("Preprocess original text file ... ")
            if self.use_sentencepiece:
                self.src = [s for s in src]
                self.tgt = [
                    [self.vocab[START_DECODING]] + t + [self.vocab[STOP_DECODING]] for t in tgt
                ]
                # sentencepiece는 oov가 없음
                self.src_oov_vocabs = [None for _ in range(len(src))]
                self.tgt_oov_vocabs = [None for _ in range(len(src))]
            else:
                # sentencepiece는 이미 index변환이 됐기때문에 pass
                self.src, self.src_oov_vocabs = zip(*[self.src2idx(s) for s in src])
                self.tgt, self.tgt_oov_vocabs = zip(
                    *[self.tgt2idx(t, s) for t, s in zip(tgt, self.src_oov_vocabs)]
                )
            with open(cached_dir, "wb") as f:
                pickle.dump(
                    (self.src, self.tgt, self.src_oov_vocabs, self.tgt_oov_vocabs,), f,
                )
                print('Save Cached file on "' + cached_dir + '"')
        assert len(self.src) == len(self.tgt)
        if max_vocab is not None and not self.use_sentencepiece:
            assert len(self.vocab) == max_vocab + len(special_tokens)
        self._idx2word = {v: k for k, v in self.vocab.items()}

    def get_vocab(self, src: List[List[str]]) -> dict:
        """
        주어진 파일로부터 단어 생성
        """
        if self.use_sentencepiece:
            vocab = self.tokenizer.vocab.token_to_idx
            # [PAD]와 [UNK]는 이미 존재
            vocab[START_DECODING] = len(vocab)
            vocab[STOP_DECODING] = len(vocab)
        else:
            vocab = {}
            for i, w in enumerate(special_tokens):
                vocab[w] = i
            ctr = Counter()
            for s in src:
                ctr.update(s)
            ctr_common = ctr.most_common(self.max_vocab if self.max_vocab is not None else None)
            for i, v in enumerate(ctr_common):
                vocab[v[0]] = i + 4
        return vocab

    def src2idx(self, src_sentence: List[str]) -> Tuple[List]:
        """
        주어진 source text로부터 단어 인덱스 리스트, OOV가 등장한 위치를 저장하는 dictionary 생성
        """
        idcs = []
        oov_vocabs = defaultdict(list)
        for i, w in enumerate(src_sentence):
            try:
                idcs.append(self.vocab[w])
                # idcs_with_oov.append(self.vocab[w])
            except KeyError:
                # vocab에 없다면 [UNK] token 반환
                idcs.append(self.vocab[UNKNOWN_TOKEN])
                # oov_vocab -> {oov_word: [idx1 where the word appears, idx2, ...,]}
                oov_vocabs[w].append(i)
        return idcs, oov_vocabs

    def tgt2idx(self, tgt_sentence: List[str], src_oov_vocabs: List[str]) -> List[int]:
        """
        주어진 target sentence(golden summary)로부터 단어 인덱스 리스트 생성
        마찬가지로 OOV가 등장한 위치를 저장하는 dictionary를 생성.
        단, source sentence에 해당 OOV가 등장했을 때만 dictonary에 저장
        """
        idcs = []
        oov_vocabs = defaultdict(list)
        for i, w in enumerate(tgt_sentence):
            try:
                idcs.append(self.vocab[w])
            except KeyError:
                idcs.append(self.vocab[UNKNOWN_TOKEN])
                if w in src_oov_vocabs:
                    # source sentence에 해당 oov단어가 등장한다면 oov_vocab에 append
                    oov_vocabs[w].append(i)

        idcs = [self.vocab[START_DECODING]] + idcs + [self.vocab[STOP_DECODING]]
        return idcs, oov_vocabs

    def idx2word(self, sentence: List[int]) -> List[str]:
        if type(sentence) == torch.Tensor:
            sentence = sentence.tolist()
        return [self._idx2word[i] for i in sentence]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx: int) -> Tuple[List]:
        return (
            self.src[idx],
            self.tgt[idx],
            self.src_oov_vocabs[idx],
            self.tgt_oov_vocabs[idx],
        )
