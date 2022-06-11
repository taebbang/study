import os

# import rouge

from glob import glob
from tqdm import tqdm
from .metric import Rouge

# from konlpy.tag import Mecab


class RougeScorer:
    def __init__(self):

        self.rouge_evaluator = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1000,
            length_limit_type="words",
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
        )

    def compute_rouge(self, ref_path, hyp_path):
        ref_fnames = glob(f"{ref_path}/*.txt")
        hyp_fnames = glob(f"{hyp_path}/*.txt")
        ref_fnames.sort()
        hyp_fnames.sort()

        print("-" * 50)
        print("# of Testset :", len(hyp_fnames))
        print("-" * 50)

        self.reference_summaries = []
        self.generated_summaries = []

        for ref_fname, hyp_fname in tqdm(zip(ref_fnames, hyp_fnames), total=len(ref_fnames)):
            assert os.path.split(ref_fname)[1] == os.path.split(hyp_fname)[1]

            with open(ref_fname, "r", encoding="utf8") as f:
                ref = f.read().replace("\n", " ")

            with open(hyp_fname, "r", encoding="utf8") as f:
                hyp = f.read().replace("\n", " ")

            # ref = " ".join(ref)
            # hyp = " ".join(hyp)

            self.reference_summaries.append(ref)
            self.generated_summaries.append(hyp)

        scores = self.rouge_evaluator.get_scores(self.generated_summaries, self.reference_summaries)
        str_scores = self.format_rouge_scores(scores)
        self.save_rouge_scores(str_scores)
        return str_scores

    def save_rouge_scores(self, str_scores):
        with open("rouge_scores.txt", "w") as output:
            output.write(str_scores)

    def format_rouge_scores(self, scores):
        return """\n
    ****** ROUGE SCORES ******
    ** ROUGE 1
    F1        >> {:.3f}
    Precision >> {:.3f}
    Recall    >> {:.3f}
    ** ROUGE 2
    F1        >> {:.3f}
    Precision >> {:.3f}
    Recall    >> {:.3f}
    ** ROUGE L
    F1        >> {:.3f}
    Precision >> {:.3f}
    Recall    >> {:.3f}""".format(
            scores["rouge-1"]["f"],
            scores["rouge-1"]["p"],
            scores["rouge-1"]["r"],
            scores["rouge-2"]["f"],
            scores["rouge-2"]["p"],
            scores["rouge-2"]["r"],
            scores["rouge-l"]["f"],
            scores["rouge-l"]["p"],
            scores["rouge-l"]["r"],
        )


# class RougeScorer:
#     def __init__(self, use_tokenizer=True):

#         self.use_tokenizer = use_tokenizer
#         if use_tokenizer:
#             self.tokenizer = Mecab()

#         self.rouge_evaluator = rouge.Rouge(
#             metrics=["rouge-n", "rouge-l"],
#             max_n=2,
#             limit_length=True,
#             length_limit=1000,
#             length_limit_type="words",
#             apply_avg=True,
#             apply_best=False,
#             alpha=0.5,  # Default F1_score
#             weight_factor=1.2,
#             stemming=True,
#         )

#     def compute_rouge(self, ref_path, hyp_path):
#         ref_fnames = glob(f"{ref_path}/*.txt")
#         hyp_fnames = glob(f"{hyp_path}/*.txt")
#         ref_fnames.sort()
#         hyp_fnames.sort()

#         reference_summaries = []
#         generated_summaries = []

#         for ref_fname, hyp_fname in tqdm(zip(ref_fnames, hyp_fnames), total=len(ref_fnames)):
#             assert os.path.split(ref_fname)[1] == os.path.split(hyp_fname)[1]

#             with open(ref_fname, "r", encoding="utf8") as f:
#                 ref = f.read().split("\n")
#                 ref = "".join(ref)

#             with open(hyp_fname, "r", encoding="utf8") as f:
#                 hyp = f.read().split("\n")
#                 hyp = "".join(hyp)

#             if self.use_tokenizer:
#                 ref = self.tokenizer.morphs(ref)
#                 hyp = self.tokenizer.morphs(hyp)

#             ref = " ".join(ref)
#             hyp = " ".join(hyp)

#             reference_summaries.append(ref)
#             generated_summaries.append(hyp)

#         scores = self.rouge_evaluator.get_scores(generated_summaries, reference_summaries)
#         str_scores = self.format_rouge_scores(scores)
#         self.save_rouge_scores(str_scores)
#         return str_scores

#     def save_rouge_scores(self, str_scores):
#         with open("rouge_scores.txt", "w") as output:
#             output.write(str_scores)

#     def format_rouge_scores(self, scores):
#         return """\n
#     ****** ROUGE SCORES ******
#     ** ROUGE 1
#     F1        >> {:.3f}
#     Precision >> {:.3f}
#     Recall    >> {:.3f}
#     ** ROUGE 2
#     F1        >> {:.3f}
#     Precision >> {:.3f}
#     Recall    >> {:.3f}
#     ** ROUGE L
#     F1        >> {:.3f}
#     Precision >> {:.3f}
#     Recall    >> {:.3f}""".format(
#             scores["rouge-1"]["f"],
#             scores["rouge-1"]["p"],
#             scores["rouge-1"]["r"],
#             scores["rouge-2"]["f"],
#             scores["rouge-2"]["p"],
#             scores["rouge-2"]["r"],
#             scores["rouge-l"]["f"],
#             scores["rouge-l"]["p"],
#             scores["rouge-l"]["r"],
#         )