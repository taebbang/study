import argparse
import dill
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TestTubeLogger  # pip install test-tube

from functools import partial
from collections import OrderedDict
from konlpy.tag import Mecab

from experiment import Experiment
from model.summarunner import SummaRunner
from model.types_ import *
from model.datasets import SumDataset, Feature, collate_fn

# from utils.preprocess import build_vocab, collate_fn


import warnings

warnings.filterwarnings(action="ignore")


def main():
    # ----------------
    # DataLoader
    # ----------------

    # data path
    train_path = args.train_path
    valid_path = args.dev_path
    vocab_path = args.vocab_path

    # vocab
    with open(vocab_path, "rb") as f:
        word_index = dill.load(f)

    # pretrained vectors

    # Feature class
    mecab = Mecab()
    feature = Feature(word_index, mecab)

    # Dataset
    trainset = SumDataset(train_path)
    validset = SumDataset(valid_path)

    # DataLoader
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, feature=feature),
        num_workers=8,
    )

    valid_loader = DataLoader(
        dataset=validset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, feature=feature),
        num_workers=8,
    )

    # SetUp Model
    # ----------------
    model = SummaRunner(vocab_size=len(word_index))
    experiment = Experiment(model, lr=0.001)

    # ----------------
    # TestTubeLogger
    # ----------------
    tt_logger = TestTubeLogger(
        save_dir=args.log_dir,
        name=args.log_name,
        debug=False,
        create_git_tag=False,
    )

    # ----------------
    # Checkpoint
    # ----------------
    checkpoint_callback = ModelCheckpoint(
        filepath=f"{args.checkpoint_dir}" + "/summarunner{epoch:02d}_{val_loss:.3f}",
        monitor="val_loss",
        verbose=True,
        save_top_k=5,
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=True)

    # ----------------
    # Trainer
    # ----------------

    trainer = Trainer(
        gpus=args.gpus,
        min_epochs=1,
        logger=tt_logger,
        num_sanity_val_steps=5,
        callbacks=[early_stopping],
        checkpoint_callback=checkpoint_callback,
        val_check_interval=args.val_interval,
        max_steps=args.steps,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.clipping,
        distributed_backend=args.distributed_backend,
        precision=args.precision,
    )

    # ----------------
    # Start Train
    # ----------------
    trainer.fit(experiment, train_loader, valid_loader)


# def main(args):
#     pl.seed_everything(args.seed)
#     model = SummaRunner(
#         vocab_path=args.vocab_path,
#         train_path=args.train_path,
#         dev_path=args.dev_path,
#         test_path=args.test_path,
#         train_batch_size=args.train_batch_size,
#         eval_batch_size=args.eval_batch_size,
#         test_batch_size=args.test_batch_size,
#         lr=args.lr,
#         embed_dim=args.embed_dim,
#         hidden_dim=args.hidden_dim,
#         pos_dim=args.embed_dim,
#         pos_num=args.embed_dim,
#         seg_num=args.embed_dim,
#         num_layers=args.embed_dim,
#         bidirectional=args.embed_dim,
#         dropout_p=args.embed_dim,
#         pretrained_vectors=args.embed_dim,
#     )
#     tt_logger = TestTubeLogger(
#         save_dir=args.log_dir,
#         name=args.log_name,
#         debug=False,
#         create_git_tag=False,
#     )

#     checkpoint_callback = ModelCheckpoint(
#         filepath=args.checkpoint_dir,
#         monitor="val_loss",
#         verbose=True,
#         save_top_k=5,
#     )
#     early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=True)
#     trainer = Trainer(
#         gpus=args.gpus,
#         min_epochs=1,
#         logger=tt_logger,
#         num_sanity_val_steps=5,
#         callbacks=[early_stopping, checkpoint_callback],
#         val_check_interval=args.val_interval,
#         max_steps=args.steps,
#         max_epochs=args.max_epochs,
#         gradient_clip_val=args.clipping,
#         distributed_backend=args.distributed_backend,
#         precision=args.precision,
#     )
#     trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for SummaRuNNer models")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Path for train dataset")
    parser.add_argument("--dev_path", type=str, required=True, help="Path for validation dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path for test dataset")
    parser.add_argument("--vocab_path", type=str, help="Path for vocabulary file")

    parser.add_argument("--seed", type=int, default=1234, help="seed for reproducibility")
    parser.add_argument("--gpus", type=int, default=2, help="the number of gpus")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoint save dir")
    parser.add_argument(
        "--load_checkpoint_from",
        type=str,
        default="./checkpoints",
        help="checkpoint file dir to load",
    )
    parser.add_argument("--log_dir", type=str, default="./logs", help="log dir")
    parser.add_argument("--log_name", type=str, default="summarunner", help="name of log files")
    parser.add_argument("--max_epochs", type=int, default=30, help="maximum epochs to train")
    parser.add_argument("--steps", help="maximum steps to train")
    parser.add_argument("--distributed_backend", type=str, default="dp")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--amp_level", type=str, default="O2")
    parser.add_argument("--lr", type=float, default=0.15, help="learning rate")
    parser.add_argument("--clipping", type=int, default=2, help="maximum gradient norm to clip")

    parser.add_argument("--train_batch_size", type=int, default=32, help="batch size at train step")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="batch size at val step")
    parser.add_argument("--test_batch_size", type=int, default=32, help="batch size at test step")

    parser.add_argument("--dropout_p", type=float, help="dropout probability")
    parser.add_argument(
        "--bidirectional", action="store_true", help="Whether to use bidirectional rnn encoders"
    )
    parser.add_argument("--num_layers", type=int, default=1, help="number of encoder rnn layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden states dimensions")
    parser.add_argument("--embed_dim", type=int, default=100, help="word embedding dimensions")
    parser.add_argument("--pos_dim", type=int, default=50, help="positional embedding dimensions")
    parser.add_argument("--pos_num", type=int, default=100, help="positional")
    parser.add_argument("--seg_num", type=int, default=25, help="word embedding dimensions")

    parser.add_argument("--val_interval", type=int, default=0.6, help="validation interval")
    args = parser.parse_args()

    main()
