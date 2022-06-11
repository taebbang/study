import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.pointergenerator import PointerGenerator


def main(args):
    if args.use_sentencepiece:
        args.vocab_size = 8000
    pl.seed_everything(args.seed)
    model = PointerGenerator(
        train_path=args.train_dataset,
        validation_path=args.val_dataset,
        test_path=args.test_dataset,
        use_sentencepiece=args.use_sentencepiece,
        vocab_path=args.vocab_path,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layer=args.num_encoder_layer,
        max_len=args.max_len,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        max_decoder_step=args.max_decoder_step,
        eval_batch_size=args.eval_batch_size,
        ptr_gen=args.ptr_gen,
        coverage=args.coverage,
        cov_loss_lambda=args.cov_loss_lambda,
    )

    checkpoints = ModelCheckpoint(dirpath=args.checkpoint_dir, save_top_k=3, monitor="val_loss")
    # logname = "pointer-generator-covloss" if args.coverage else "pointer-generator-no-covloss"
    tb_logger = TensorBoardLogger(save_dir=args.log_dir)
    early_stop_callback = EarlyStopping()

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_steps=args.steps,
        val_check_interval=args.val_interval,
        num_sanity_val_steps=0,
        logger=tb_logger,
        gradient_clip_val=args.clipping,
        distributed_backend=args.distributed_backend,
        precision=args.precision,
        checkpoint_callback=checkpoints,
        resume_from_checkpoint=args.load_checkpoint_from,
        # early_stop_callback=early_stop_callback,
        # deterministic=True,
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, required=True, help="Path for train dataset")
    parser.add_argument(
        "--val_dataset", type=str, required=True, help="Path for validation dataset"
    )
    parser.add_argument("--test_dataset", type=str, required=True, help="Path for test dataset")
    parser.add_argument("--vocab_path", type=str, help="Path for vocabulary file")
    parser.add_argument(
        "--use_sentencepiece", action="store_true", help="Whether to use sentencepiece tokenizer"
    )

    parser.add_argument("--seed", type=int, default=1234, help="seed for reproducibility")
    parser.add_argument("--gpus", type=int, default=2, help="the number of gpus")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoint save dir")
    parser.add_argument("--load_checkpoint_from", type=str, help="checkpoint file dir to load")
    parser.add_argument("--log_dir", type=str, help="log dir")
    parser.add_argument("--steps", type=int, default=100000, help="maximum steps to train")
    parser.add_argument("--distributed_backend", type=str, default="ddp")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--amp_level", type=str, default="O2")
    parser.add_argument("--lr", type=float, default=0.15, help="learning rate")
    parser.add_argument(
        "--lr_coverage", type=float, default=0.15, help="learning rate for coverage"
    )
    parser.add_argument("--train_batch_size", type=int, default=55, help="batch size at train step")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="batch size at val step")
    parser.add_argument("--max_len", type=int, default=400, help="maximum length of sequence")
    parser.add_argument("--clipping", type=int, default=2, help="maximum gradient norm to clip")

    parser.add_argument("--ptr_gen", action="store_true", help="activate pointer generator")
    parser.add_argument("--coverage", action="store_true", help="activate coverage mechanism")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden states dimensions")
    parser.add_argument("--embed_dim", type=int, default=128, help="word embedding dimensions")
    parser.add_argument("--vocab_size", type=int, default=50000, help="number of vocab")
    parser.add_argument(
        "--num_encoder_layer", type=int, default=1, help="number of encoder rnn layers"
    )
    parser.add_argument("--cov_loss_lambda", type=float, default=1.0, help="number of vocab")

    parser.add_argument("--val_interval", type=int, default=400, help="validation interval")
    parser.add_argument(
        "--max_decoder_step",
        type=int,
        default=100,
        help="max length for sentence created by decoder",
    )
    parser.add_argument("--beam_size", type=int, default=4, help="beam search size")

    args = parser.parse_args()
    main(args)
