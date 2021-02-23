import os
import pickle
from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers.modeling_bert import BertPreTrainedModel

from dataset import TokenizedDataset


bert_config = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "type_vocab_size": 2,
    "vocab_size": 8002,
}


class ContentSelector(pl.LightningModule):
    """ Sequence Tagging Model for Content Selection """

    def __init__(
        self,
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        lr: float = None,
        warmup_percent: float = 0.1,
        train_batch_size: int = None,
        val_batch_size: int = None,
        num_classes: int = 2,
        num_workers: int = 2,
        gpus: int = 2,
        config: dict = bert_config,
    ) -> None:
        super(ContentSelector, self).__init__()
        self.save_hyperparameters()
        self.lr = self.hparams.lr
        self.lr_scale = 0
        self.loss = nn.CrossEntropyLoss()

        self.bert, self.vocab = get_pytorch_kobert_model()
        self.dropout = nn.Dropout(self.hparams.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.hparams.config["hidden_size"], num_classes)

    def forward(self, batch):
        hidden, _ = self.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pooled = self.dropout(hidden)
        logits = self.classifier(pooled)
        return logits

    def train_dataloader(self) -> DataLoader:
        self.train_dset = TokenizedDataset(self.hparams.train_path)
        return DataLoader(
            self.train_dset,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        self.val_dset = TokenizedDataset(self.hparams.val_path)
        return DataLoader(
            self.val_dset,
            batch_size=self.hparams.val_batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        self.test_dset = TokenizedDataset(self.hparams.test_path)
        return DataLoader(
            self.test_dset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def configure_optimizers(self) -> optim:
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        active_loss = attention_mask.view(-1)
        active_logits = logits.view(-1, self.hparams.num_classes)
        active_labels = torch.where(
            active_loss,
            labels.view(-1),
            torch.tensor(self.loss.ignore_index).type_as(labels),
        )

        loss = self.loss(active_logits, active_labels)
        # tensorboard_logs = {
        #     "train_loss": loss,
        #     "learning_rate": self.lr_scale * self.lr,
        # }
        self.log("train_loss", loss)
        self.log("learning_rate", self.lr_scale * self.lr)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        active_loss = attention_mask.view(-1)
        active_logits = logits.view(-1, self.hparams.num_classes)
        active_labels = torch.where(
            active_loss,
            labels.view(-1),
            torch.tensor(self.loss.ignore_index).type_as(labels),
        )
        loss = self.loss(active_logits, active_labels)
        return loss

    def validation_epoch_end(self, outputs):
        loss = torch.stack([output for output in outputs], 0).mean()
        # tensorboard_logs = {"val_loss": loss}
        self.log("val_loss", loss)
        # return {
        #     "avg_val_loss": loss,
        #     "log": tensorboard_logs,
        # }

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        active_loss = attention_mask.view(-1)
        active_logits = logits.view(-1, self.hparams.num_classes)
        active_labels = torch.where(
            active_loss,
            labels.view(-1),
            torch.tensor(self.loss.ignore_index).type_as(labels),
        )

        loss = self.loss(active_logits, active_labels)

        prob = logits.softmax(-1)
        pred = prob[..., 1]
        pred = pred * attention_mask

        return {"test_loss": loss, "pred": pred}

    def test_epoch_end(self, outputs):
        preds = torch.cat([output["pred"] for output in outputs])
        self.save_preds(preds)
        loss = torch.stack([output["test_loss"] for output in outputs], 0).mean()
        return {
            "avg_test_loss": loss,
        }

    def save_preds(self, preds):
        dirs, ext = os.path.splitext(self.hparams.test_path)
        save_dir = dirs + "_cs" + ext
        with open(save_dir, "wb") as f:
            pickle.dump(preds, f)
