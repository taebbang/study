import argparse
import pickle
from model.pointergenerator import PointerGenerator
from model.dataset import TextDataset
from model.generator import SentenceDecoder
from torch.utils.data import DataLoader, Dataset
from model.model_utils import Collator
import pytorch_lightning as pl
import torch
import os
import json

try:
    from metric import Rouge
except:
    print("Try pip install py-rouge")


PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[UNK]"
START_DECODING = "[START]"
STOP_DECODING = "[STOP]"


def main(args):
    device = "cuda" if args.use_gpu else "cpu"
    val_dset = TextDataset(
        args.test_dataset_path,
        mode="eval",
        vocab_dir=args.vocab_dir,
        use_sentencepiece=args.use_sentencepiece,
    )
    if args.use_sentencepiece:
        vocab_ = val_dset.vocab
    collate_fn = Collator(
        val_dset, args.ptr_gen, args.max_len, args.max_decoder_step, args.use_sentencepiece
    )
    dloader = DataLoader(
        val_dset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=16, shuffle=False,
    )
    if args.content_selection_path is not None:
        with open(args.content_selection_path, "rb") as f:
            selected_contents = pickle.load(f)
        selected_contents = torch.tensor(selected_contents)[:, 1:].to(device)  # remove [CLS] token
        selected_contents = selected_contents.split(args.batch_size)

    model = PointerGenerator.load_from_checkpoint(args.checkpoint_path).to(device)
    decoder = SentenceDecoder(model)
    output = []
    tgt_total = []
    vocab_total = []
    for i, batch in enumerate(dloader):
        (
            src,
            tgt_input,
            tgt_output_oov,
            src_len,
            tgt_len,
            src_padding_mask,
            tgt_padding_mask,
            src_oov,
        ) = [i.to(device) for i in batch[:-1]]

        vocab = batch[-1]
        vocab_total.append(vocab)
        tgt_total += tgt_output_oov

        if args.content_selection_path is not None:
            contents = selected_contents[i]
            if contents.shape[1] != src.shape[1]:
                contents = contents[:, : src.shape[1]]

        else:
            contents = None
        output += decoder.generate(
            src=src,
            src_len=src_len,
            src_padding_mask=src_padding_mask,
            vocab=vocab if not args.use_sentencepiece else vocab_,
            src_oov=src_oov,
            min_length=5,
            num_beams=4,
            num_return_sequences=1,
            do_sample=False,
            early_stopping=False,
            temperature=1,
            top_k=0,
            top_p=1.0,
            contents=contents,
            length_penalty=args.length_penalty,
            copy_penalty_beta=args.copy_penalty_beta,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            contents_threshold=args.contents_threshold
        )

    tgt_kor = []
    out_kor = []

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'output.txt'), 'w') as f_out:
        with open(os.path.join(args.output_path, 'tgt.txt'), 'w') as f_tgt:
            for i in range(len(dloader)):
                out = output[i * args.batch_size : (i + 1) * args.batch_size]
                tgt = tgt_total[i * args.batch_size : (i + 1) * args.batch_size]
                vocab = vocab_total[i] if not args.use_sentencepiece else vocab_
                idx2word = {v: k for k, v in vocab.items()}

                for b in range(len(out)):
                    tgt_ = [
                        idx2word[i.item()].split("_")[0]
                        for i in tgt[b]
                        if i not in [vocab[PAD_TOKEN], vocab[START_DECODING], vocab[STOP_DECODING]]
                    ]
                    out_ = [
                        idx2word[i.item()].split("_")[0]
                        for i in out[b]
                        if i not in [vocab[PAD_TOKEN], vocab[START_DECODING], vocab[STOP_DECODING]]
                    ]
                    if args.use_sentencepiece:
                        tgt_ = "".join(tgt_).replace("▁", " ")
                        out_ = "".join(out_).replace("▁", " ")
                    else:
                        tgt_ = " ".join(tgt_)
                        out_ = " ".join(out_)

                    tgt_kor.append(tgt_)
                    out_kor.append(out_)
                    
                    f_out.write(out_)
                    f_out.write('\n')
                    f_tgt.write(tgt_)
                    f_tgt.write('\n')

    rouge_evaluator = Rouge(
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
    scores = rouge_evaluator.get_scores(out_kor, tgt_kor)
    str_scores = format_rouge_scores(scores)
    print(str_scores)
    os.makedirs(args.rouge_path, exist_ok=True)
    save_rouge_scores(args.rouge_path, str_scores)

    with open(args.test_dataset_path, "r", encoding="utf-8") as f:
        jsonl = list(f)
    data = []
    for json_str in jsonl:
        data.append(json.loads(json_str))
    src = [d["article_original"] for d in data]
    src = [" ".join(s) for s in src]
    
    with open(os.path.join(args.output_path, 'src.txt'), 'w') as f_src:
        for s in src:
            f_src.write(s)
            f_src.write('\n')
    # with open(os.path.join(args.output_path, 'output.txt'), 'w') as f:
    #     for o in out_kor:
    #         f.write(o)
    #         f.write('\n')


def format_rouge_scores(scores):
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


def save_rouge_scores(path, str_scores):
    with open(os.path.join(path, "rouge_scores.txt"), "w") as output:
        output.write(str_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_dataset_path", type=str, required=True, help="Path for dataset to inference"
    )
    parser.add_argument(
        "--vocab_dir", type=str, required=True, help="Directory for vocab dictionary"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint to use")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--rouge_path", type=str, default="./", help="Path for rouge score")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use gpu")
    parser.add_argument(
        "--use_sentencepiece", action="store_true", help="Whether to use sentencepiece tokenizer"
    )
    parser.add_argument("--ptr_gen", action="store_true", help="activate pointer generator")
    parser.add_argument("--max_len", type=int, default=400, help="maximum length of sequence")
    parser.add_argument(
        "--max_decoder_step",
        type=int,
        default=200,
        help="max length for sentence created by decoder",
    )

    # for bottom up summarization
    parser.add_argument(
        "--content_selection_path", type=str, help="Path for bottom up summarization"
    )
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--copy_penalty_beta", type=float, default=0.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--contents_threshold", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
