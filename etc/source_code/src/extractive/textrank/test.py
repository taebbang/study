import os
import argparse

from tqdm import tqdm
from textrank import TextRank
from utils.data import get_data
from utils.types_ import *


# Parser
parser = argparse.ArgumentParser(description="Extractive Summarization using TextRank")

# model
parser.add_argument(
    "--min_count",
    type=int,
    default=2,
    help="Minumum frequency of words will be used to construct sentence graph",
)
parser.add_argument(
    "--min_sim",
    type=float,
    default=0.3,
    help="Minimum similarity of sents or words will be used to construct sentence graph",
)
parser.add_argument(
    "--tokenizer", type=str, default="mecab", help="Tokenizer for korean, default is mecab"
)
parser.add_argument("--noun", type=bool, default=False, help="option for using just nouns")
parser.add_argument(
    "--similarity",
    type=str,
    default="cosine",
    help="similarity type to use choose cosine or textrank",
)
parser.add_argument("--df", type=float, default=0.85, help="PageRank damping factor")
parser.add_argument("--max_iter", type=int, default=50, help="Number of PageRank iterations")
parser.add_argument("--method", type=str, default="algebraic", help="Method of PageRank")
parser.add_argument("--topk", type=int, default=3, help="Number of sentences/words to summarize")
# data
parser.add_argument("--test_path", type=str, help="Data path to load")
# output
parser.add_argument("--output_path", type=str, required=True, help="directory for results")

args = parser.parse_args()


if __name__ == "__main__":

    # initialize Textrank
    model = TextRank(
        min_count=args.min_count,
        min_sim=args.min_sim,
        tokenizer=args.tokenizer,
        noun=args.noun,
        similarity=args.similarity,
        df=args.df,
        method=args.method,
        stopwords=None,
    )

    data = get_data(args.test_path)

    output_path = args.output_path
    hyp_path = f"{output_path}/hyp"
    abs_ref_path = f"{output_path}/abs_ref"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(hyp_path):
        os.makedirs(hyp_path)
    if not os.path.exists(abs_ref_path):
        os.makedirs(abs_ref_path)

    for articles in tqdm(data):
        doc_id, sents, gold = articles

        hyp = model.summarize(sents, args.topk)

        with open(f"{abs_ref_path}/{doc_id}.txt", "w", encoding="utf8") as f:
            f.write(gold)
        with open(f"{hyp_path}/{doc_id}.txt", "w", encoding="utf8") as f:
            f.write(hyp)