import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from net import ContentSelector


def main(args):
    pl.seed_everything(args.seed)
    tb_logger = TensorBoardLogger(save_dir="./logs", name="content-selector")
    checkpoints = ModelCheckpoint(filepath=args.checkpoint_dir)  # , save_top_k=3)

    if args.mode == "train":
        model = ContentSelector(
            train_path=args.train_path,
            val_path=args.val_path,
            lr=args.lr,
            warmup_percent=args.warmup_percent,
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            num_classes=args.num_classes,
            num_workers=args.num_workers,
            gpus=args.gpus,
        )

        trainer = pl.Trainer(
            gpus=args.gpus,
            max_epochs=args.epoch,
            logger=tb_logger,
            val_check_interval=100,
            checkpoint_callback=checkpoints,
            # early_stop_callback=True,
            # deterministic=True,
            distributed_backend=args.distributed_backend,
            precision=args.precision,
        )
        trainer.fit(model)
    else:
        model = ContentSelector(
            test_path=args.test_path,
            val_batch_size=args.val_batch_size,
            lr=args.lr,
            num_classes=args.num_classes,
            num_workers=args.num_workers,
        )
        trainer = pl.Trainer(
            gpus=args.gpus,
            resume_from_checkpoint=args.load_checkpoint_from,
            distributed_backend="dp",
        )
        print("Test Phase...")
        preds = trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--val_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument(
        "--gpus", type=int, default=2, help="-1 means all gpus available"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--distributed_backend", type=str, default="ddp")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--mode", type=str, default="train", help="train/inference")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoint save dir")
    parser.add_argument(
        "--load_checkpoint_from", type=str, help="checkpoint file dir to load"
    )
    parser.add_argument(
        "--path", type=str, default="../../../datasets/contentselection/"
    )
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_percent", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--precision", type=int, default=16)

    args = parser.parse_args()
    main(args)