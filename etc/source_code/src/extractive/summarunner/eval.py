import argparse
from utils import RougeScorer
import os


# Paser
parser = argparse.ArgumentParser(description="Calculate ROUGE Score")

parser.add_argument("--ref_path", type=str, default="./outputs/abs_ref", help="Path of References")
parser.add_argument("--hyp_path", type=str, default="./outputs/hyp", help="Path of Hypothesis")
parser.add_argument("--result_path", type=str, default="./results/", help="Path of rouge scores")

args = parser.parse_args()

if __name__ == "__main__":
    # ref_path = "./outputs/abs_ref"
    # hyp_path = "./outputs/hyp"

    ref_path = args.ref_path
    hyp_path = args.hyp_path

    rouge_eval = RougeScorer()
    result = rouge_eval.compute_rouge(ref_path, hyp_path)

    os.makedirs(args.result_path, exist_ok=True)
    with open(os.path.join(args.result_path, "rouge_scores.txt"), "w") as output:
        output.write(result)

    print(result)