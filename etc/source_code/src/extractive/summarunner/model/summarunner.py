import numpy as np
import dill
from functools import partial
from konlpy.tag import Mecab
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from torch.autograd import Variable

from .model_utils import Encoder, avg_pool1d
from .datasets import SumDataset, Feature


# Device configuration
# DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


class SummaRunner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        pos_dim: int = 50,
        pos_num: int = 100,
        seg_num: int = 25,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_p: float = 0.3,
        pretrained_vectors: np.ndarray = None,
    ):
        super(SummaRunner, self).__init__()
        self.hidden_dim = hidden_dim
        self.abs_pos_embed = nn.Embedding(pos_num, pos_dim)  # absolute postion
        self.rel_pos_embed = nn.Embedding(seg_num, pos_dim)  # relative position

        self.encoder = Encoder(
            vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout_p
        )

        self.fc = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        # Parameters of Classification Layer
        # P(y_j = 1|h_j, s_j, d), Eq.6 in SummaRuNNer paper
        self.content = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.salience = nn.Bilinear(2 * hidden_dim, 2 * hidden_dim, 1, bias=False)
        self.novelty = nn.Bilinear(2 * hidden_dim, 2 * hidden_dim, 1, bias=False)
        self.abs_pos = nn.Linear(pos_dim, 1, bias=False)
        self.rel_pos = nn.Linear(pos_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

    def forward(self, docs, doc_lens, max_doc_len):
        sent_out = self.encoder(docs, doc_lens, max_doc_len)
        docs = avg_pool1d(sent_out, doc_lens)

        probs = []
        for index, doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index, :doc_len, :]
            doc = torch.tanh(self.fc(docs[index])).unsqueeze(0)
            s = Variable(torch.zeros(1, 2 * self.hidden_dim)).type_as(docs)
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]])).type_as(docs)
                abs_index = abs_index.type(torch.long)
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)

                rel_index = int(round((position + 1) * 9.0 / doc_len.item()))
                rel_index = Variable(torch.LongTensor([[rel_index]])).type_as(docs)
                rel_index = rel_index.type(torch.long)
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)

                # classification layer
                content = self.content(h)
                salience = self.salience(h, doc)
                novelty = -1 * self.novelty(h, torch.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                # P(y_j = 1|h_j, s_j, d) Eq.6 in SummaRuNNer paper
                prob = torch.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob, h)
                probs.append(prob)

        return torch.cat(probs).squeeze()


# class SummaRunner(pl.LightningModule):
#     def __init__(
#         self,
#         vocab_path,
#         train_path,
#         dev_path,
#         test_path,
#         train_batch_size,
#         eval_batch_size,
#         test_batch_size,
#         lr,
#         embed_dim: int = 100,
#         hidden_dim: int = 128,
#         pos_dim: int = 50,
#         pos_num: int = 100,
#         seg_num: int = 25,
#         num_layers: int = 1,
#         bidirectional: bool = True,
#         dropout_p: float = 0.3,
#         pretrained_vectors: np.ndarray = None,
#     ):
#         super(SummaRunner, self).__init__()
#         self.save_hyperparameters()
#         # vocab & feature class
#         with open(self.vocab_path, "rb") as f:
#             word_index = dill.load(f)
#         self.vocab_size = len(word_index)
#         tokenizer = Mecab()
#         self.feature = Feature(word_index, tokenizer)

#         self.abs_pos_embed = nn.Embedding(
#             self.pos_num, self.pos_dim
#         )  # absolute postion
#         self.rel_pos_embed = nn.Embedding(
#             self.seg_num, self.pos_dim
#         )  # relative position

#         self.encoder = Encoder(
#             vocab_size=self.vocab_size,
#             embed_dim=self.embed_dim,
#             hidden_dim=self.hidden_dim,
#             num_layers=self.num_layers,
#             bidirectional=self.bidirectional,
#             dropout_p=self.dropout_p,
#         )

#         self.fc = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)

#         # Parameters of Classification Layer
#         # P(y_j = 1|h_j, s_j, d), Eq.6 in SummaRuNNer paper
#         self.content = nn.Linear(2 * self.hidden_dim, 1, bias=False)
#         self.salience = nn.Bilinear(
#             2 * self.hidden_dim, 2 * self.hidden_dim, 1, bias=False
#         )
#         self.novelty = nn.Bilinear(
#             2 * self.hidden_dim, 2 * self.hidden_dim, 1, bias=False
#         )
#         self.abs_pos = nn.Linear(self.pos_dim, 1, bias=False)
#         self.rel_pos = nn.Linear(self.pos_dim, 1, bias=False)
#         self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

#         self.loss_fn = nn.BCELoss()

#     def forward(self, docs, doc_lens, max_doc_len):
#         sent_out = self.encoder(docs, doc_lens, max_doc_len)
#         docs = avg_pool1d(sent_out, doc_lens)

#         probs = []
#         for index, doc_len in enumerate(doc_lens):
#             valid_hidden = sent_out[index, :doc_len, :]
#             doc = torch.tanh(self.fc(docs[index])).unsqueeze(0)
#             s = Variable(torch.zeros(1, 2 * self.hidden_dim)).type_as(docs)
#             for position, h in enumerate(valid_hidden):
#                 h = h.view(1, -1)
#                 # get position embeddings
#                 abs_index = Variable(torch.LongTensor([[position]])).type_as(docs)
#                 abs_index = abs_index.type(torch.long)
#                 abs_features = self.abs_pos_embed(abs_index).squeeze(0)

#                 rel_index = int(round((position + 1) * 9.0 / doc_len.item()))
#                 rel_index = Variable(torch.LongTensor([[rel_index]])).type_as(docs)
#                 rel_index = rel_index.type(torch.long)
#                 rel_features = self.rel_pos_embed(rel_index).squeeze(0)

#                 # classification layer
#                 content = self.content(h)
#                 salience = self.salience(h, doc)
#                 novelty = -1 * self.novelty(h, torch.tanh(s))
#                 abs_p = self.abs_pos(abs_features)
#                 rel_p = self.rel_pos(rel_features)
#                 # P(y_j = 1|h_j, s_j, d) Eq.6 in SummaRuNNer paper
#                 prob = torch.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
#                 s = s + torch.mm(prob, h)
#                 probs.append(prob)

#         return torch.cat(probs).squeeze()

#     def accuracy(self, preds, labels):
#         preds = torch.round(preds)
#         corrects = (preds == labels).float().sum()
#         acc = corrects / labels.numel()
#         return acc

#     def training_step(self, batch, batch_idx):
#         docs, targets, doc_lens, max_doc_len, _, _, _ = batch

#         preds = self.forward(docs, doc_lens, max_doc_len)

#         labels = []
#         for idx, doc_len in enumerate(doc_lens):
#             label = targets[idx][:doc_len]
#             labels.append(label)
#         labels = torch.cat(labels, dim=0)

#         train_loss = self.loss_fn(preds, labels)
#         train_acc = self.accuracy(preds, labels)
#         log_dict = {"train_acc": train_acc.detach(), "train_loss": train_loss.detach()}

#         output = OrderedDict(
#             {
#                 "loss": train_loss,
#                 "progress_bar": {"train_acc": train_acc},
#                 "log": log_dict,
#             }
#         )
#         return output

#     def validation_step(self, batch, batch_idx):
#         docs, targets, doc_lens, max_doc_len, _, _, _ = batch

#         preds = self.forward(docs, doc_lens, max_doc_len)

#         labels = []
#         for idx, doc_len in enumerate(doc_lens):
#             label = targets[idx][:doc_len]
#             labels.append(label)
#         labels = torch.cat(labels, dim=0)

#         val_loss = self.loss_fn(preds, labels)
#         val_acc = self.accuracy(preds, labels)

#         tqdm_dict = {"val_acc": val_acc.detach(), "val_loss": val_loss.detach()}
#         output = OrderedDict(
#             {
#                 "val_loss": val_loss,
#                 "val_acc": val_acc,
#                 "log": tqdm_dict,
#                 "progress_bar": tqdm_dict,
#             }
#         )
#         return output

#     def validation_epoch_end(self, outputs):
#         val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
#         val_acc_mean = torch.stack([x["val_acc"] for x in outputs]).mean()
#         return {"val_loss": val_loss_mean.detach(), "val_acc": val_acc_mean.detach()}

#     def test_step(self, batch, batch_idx):
#         docs, targets, doc_lens, max_doc_len, _, _, _ = batch

#         preds = self.forward(features, doc_lens, max_doc_len)
#         test_acc = self.accuracy(preds, targets)
#         return {
#             "test_acc": test_acc,
#         }

#     def test_epoc_end(self, outputs):
#         test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
#         return {"test_loss": test_loss_mean.detach()}

#     def collate_fn(self, batch):
#         docs = [entry[0] for entry in batch]
#         labels_list = [entry[1] for entry in batch]
#         summaries_list = [entry[2] for entry in batch]

#         features, targets, doc_lens, ext_sums, abs_sums, docs = self.feature.make_features(
#             docs, labels_list, summaries_list
#         )

#         docs = []
#         labels = []
#         start = 0
#         pad_dim = len(features[0])
#         max_doc_len = max(doc_lens)
#         for doc_len in doc_lens:
#             stop = start + doc_len
#             doc = features[start:stop]
#             target = targets[start:stop]
#             start = stop

#             doc = torch.LongTensor(doc)
#             if len(doc) == max_doc_len:
#                 docs.append(doc.unsqueeze(0))
#             else:
#                 pad = torch.zeros(max_doc_len - doc_len, pad_dim, dtype=torch.long)
#                 docs.append(torch.cat([doc, pad]).unsqueeze(0))

#             if len(target) == max_doc_len:
#                 labels.append(torch.FloatTensor(target).unsqueeze(0))
#             else:
#                 pad = torch.zeros(max_doc_len - doc_len)
#                 target = torch.FloatTensor(target)
#                 labels.append(torch.cat([target, pad]).unsqueeze(0))

#         docs = torch.cat(docs, dim=0)
#         labels = torch.cat(labels, dim=0)
#         targets = torch.FloatTensor(targets)
#         doc_lens = torch.LongTensor(doc_lens)
#         return docs, labels, doc_lens, max_doc_len, ext_sums, abs_sums, docs

#     def train_dataloader(self):
#         trainset = SumDataset(self.train_path)
#         train_loader = DataLoader(
#             dataset=trainset,
#             batch_size=self.train_batch_size,
#             shuffle=True,
#             collate_fn=self.collate_fn,
#         )
#         return train_loader

#     def val_dataloader(self):
#         devset = SumDataset(self.dev_path)
#         dev_loader = DataLoader(
#             dataset=devset,
#             batch_size=self.eval_batch_size,
#             shuffle=False,
#             collate_fn=self.collate_fn,
#         )
#         return dev_loader

#     def test_dataloader(self):
#         testset = SumDataset(self.test_path)
#         test_loader = DataLoader(
#             dataset=testset,
#             batch_size=self.test_batch_size,
#             shuffle=False,
#             collate_fn=self.collate_fn,
#         )
#         return test_loader

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
