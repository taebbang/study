import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from model.types_ import *

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Experiment(pl.LightningModule):
    def __init__(self, model, lr):
        super(Experiment, self).__init__()

        self.model = model
        self.lr = lr
        self._loss = nn.BCELoss()

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, docs, doc_lens, max_doc_len):
        return self.model(docs, doc_lens, max_doc_len)

    def loss_fn(self, preds, labels):
        bce_loss = self._loss(preds, labels)
        return bce_loss

    def accuracy(self, preds, labels):
        preds = torch.round(preds)
        corrects = (preds == labels).float().sum()
        acc = corrects / labels.numel()
        return acc

    def training_step(self, batch, batch_idx):
        docs, targets, doc_lens, max_doc_len, _, _, _, _ = batch

        preds = self.forward(docs, doc_lens, max_doc_len)

        labels = []
        for idx, doc_len in enumerate(doc_lens):
            label = targets[idx][:doc_len]
            labels.append(label)
        labels = torch.cat(labels, dim=0)

        train_loss = self.loss_fn(preds, labels)
        train_acc = self.accuracy(preds, labels)
        log_dict = {"train_acc": train_acc.detach(), "train_loss": train_loss.detach()}

        output = OrderedDict(
            {
                "loss": train_loss,
                "progress_bar": {"train_acc": train_acc},
                "log": log_dict,
            }
        )
        return output

    def validation_step(self, batch, batch_idx):
        docs, targets, doc_lens, max_doc_len, _, _, _, _ = batch

        preds = self.forward(docs, doc_lens, max_doc_len)

        labels = []
        for idx, doc_len in enumerate(doc_lens):
            label = targets[idx][:doc_len]
            labels.append(label)
        labels = torch.cat(labels, dim=0)

        val_loss = self.loss_fn(preds, labels)
        val_acc = self.accuracy(preds, labels)

        tqdm_dict = {"val_acc": val_acc.detach(), "val_loss": val_loss.detach()}
        output = OrderedDict(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "log": tqdm_dict,
                "progress_bar": tqdm_dict,
            }
        )
        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["val_acc"] for x in outputs]).mean()
        return {"val_loss": val_loss_mean.detach(), "val_acc": val_acc_mean.detach()}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
