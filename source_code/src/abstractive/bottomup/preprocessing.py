# Content Selector 학습용 데이터 제작 코드 (Korean)
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from typing import List
from itertools import chain

import gluonnlp as nlp
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from tqdm import tqdm


def load_data(pth: str, mode: str) -> List[dict]:
    path = os.path.join(pth, f"{mode}.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        jsonl = list(f)
    data = []
    for json_str in jsonl:
        data.append(json.loads(json_str))

    if mode == "train":
        return data[:100000]
    return data


def compile_substring(start: int, end: int, subsequence: List[str]) -> str:
    if start == end:
        return subsequence[start]
    return " ".join(subsequence[start : end + 1])


def make_aux_tgt(s: List[str], t: List[str]) -> List[int]:
    start, end = 0, 0
    matches = []
    matchstrings = Counter()

    while end < len(s):
        currentseq = compile_substring(start, end, s)
        if currentseq in t and end < len(s) - 1:
            end += 1
        else:
            if start >= end:
                matches.extend(["0"] * (end - start + 1))
                end += 1
            else:
                full_string = compile_substring(start, end - 1, s)
                if matchstrings[full_string] >= 1:
                    matches.extend(["0"] * (end - start))
                else:
                    matches.extend(["1"] * (end - start))
                    matchstrings[full_string] += 1
            start = end

    return list(map(int, " ".join(matches).split()))


def tokenize(data, tok):
    return {
        "orig_src": data["article_original"],
        "orig_abs": [data["abstractive"]],
        "tokenized_src": [tok(t) for t in data["article_original"]],
        "tokenized_abs": [tok(t) for t in [data["abstractive"]]],
    }


def main(args):
    root = args.path
    mode = args.mode
    dset = load_data(root, mode)

    _, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    tokenized_dset = []
    start_time = time.time()
    for d in tqdm(dset):
        tokenized_dset.append(tokenize(d, tok))
    print("--- %s seconds for tokenizing ---" % (time.time() - start_time))

    start_time = time.time()
    result = {"token": [], "tgt": []}
    for idx, data in tqdm(enumerate(tokenized_dset)):
        src = " ".join([" ".join(d) for d in data["tokenized_src"]]).split(" ")
        tgt = " ".join([" ".join(d) for d in data["tokenized_abs"]]).split(" ")

        auxiliary_tgt = make_aux_tgt(src, tgt)

        assert len(src) == len(
            auxiliary_tgt
        ), f"Length mismatch: {len(src)}, {len(auxiliary_tgt)}"

        result["token"].append(src)
        result["tgt"].append(auxiliary_tgt)

    print("--- %s seconds for generating labels ---" % (time.time() - start_time))

    with open(f"{args.save_path}/contentselection_{mode}.pickle", "wb") as f:
        pickle.dump(result, f)
    print("--- Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="../../../datasets/kor_data/total_data/"
    )
    parser.add_argument(
        "--save_path", type=str, default="../../../datasets/contentselection/"
    )
    parser.add_argument("--mode", type=str, default="train", help="train/dev/test")

    args = parser.parse_args()
    main(args)
