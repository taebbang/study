import os
import argparse
import dill
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from functools import partial
from collections import OrderedDict
from konlpy.tag import Mecab
from tqdm import tqdm

from experiment import Experiment
from model import SummaRunner
from model import SumDataset, Feature
from model import collate_fn
from model.types_ import *


DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Test for SummaRuNNer models")
parser.add_argument("--test_path", type=str, required=True, help="Path for test dataset")
parser.add_argument("--vocab_path", type=str, required=True, help="Path for vocabulary file")
parser.add_argument("--test_batch_size", type=int, default=32, help="Batch size for test")
parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint for summarunner model")
parser.add_argument("--topk", type=int, default=3, help="number of sentences")
parser.add_argument("--output_path", type=str, required=True, help="directory for results")

args = parser.parse_args()


def main():
    # ----------------
    # DataLoader
    # ----------------

    # data path
    test_path = args.test_path
    vocab_path = args.vocab_path

    # vocab
    with open(vocab_path, "rb") as f:
        word_index = dill.load(f)

    # Feature class
    feature = Feature(word_index, Mecab())

    # Dataset
    testset = SumDataset(test_path)

    # DataLoader
    test_loader = DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, feature=feature),
        num_workers=8,
    )

    # -------------------------
    # Pre-trained Weights Load
    # -------------------------

    ckpt_path = args.ckpt_path

    checkpoint = torch.load(ckpt_path)
    checkpoint["state_dict"] = OrderedDict(
        [(key.replace("model.", ""), val) for key, val in checkpoint["state_dict"].items()]
    )

    # ----------------
    # SetUp Model
    # ----------------
    model = SummaRunner(vocab_size=len(word_index)).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # -------------------------
    # Create Ouput Directory
    # -------------------------
    output_path = args.output_path  # "./outputs"
    hyp_path = f"{output_path}/hyp"
    abs_ref_path = f"{output_path}/abs_ref"
    ext_ref_path = f"{output_path}/ext_ref"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(hyp_path):
        os.makedirs(hyp_path)
    if not os.path.exists(abs_ref_path):
        os.makedirs(abs_ref_path)
    if not os.path.exists(ext_ref_path):
        os.makedirs(ext_ref_path)

    # ----------------
    # Model Test
    # ----------------

    num_topk = args.topk
    batch_size = args.test_batch_size
    # file_id = 1
    for batch in tqdm(test_loader, total=len(testset) // batch_size):
        docs, labels, doc_lens, max_doc_len, ext_sums, abs_sums, orgin_docs, doc_ids = batch
        preds = model(docs.to(DEVICE), doc_lens, max_doc_len)

        start = 0
        for d_idx, doc_len in enumerate(doc_lens):
            stop = start + doc_len
            pred = preds[start:stop]

            topk_indices = pred.topk(num_topk)[1].tolist()
            topk_indices.sort()

            doc = orgin_docs[d_idx]
            doc_id = doc_ids[d_idx]
            hyp = [doc[idx] for idx in topk_indices]
            ext_ref = ext_sums[d_idx]
            abs_ref = abs_sums[d_idx]

            with open(f"{ext_ref_path}/{doc_id}.txt", "w", encoding="utf8") as f:
                f.write("\n".join(ext_ref))
            with open(f"{abs_ref_path}/{doc_id}.txt", "w", encoding="utf8") as f:
                f.write(abs_ref)
            with open(f"{hyp_path}/{doc_id}.txt", "w", encoding="utf8") as f:
                f.write("\n".join(hyp))

            start = stop
            # file_id += 1


if __name__ == "__main__":
    main()