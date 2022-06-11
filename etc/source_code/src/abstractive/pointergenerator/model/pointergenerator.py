import os
import sys
from typing import List, Tuple

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import TextDataset
from .model_utils import Attention, Decoder, Encoder, Collator

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[UNK]"
START_DECODING = "[START]"
STOP_DECODING = "[STOP]"

special_tokens = [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]


class PointerGenerator(pl.LightningModule):
    """Pointer-generator Network"""

    def __init__(
        self,
        train_path,
        validation_path,
        use_sentencepiece,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_encoder_layer,
        max_len,
        max_decoder_step,
        lr,
        train_batch_size,
        eval_batch_size,
        ptr_gen,
        coverage,
        cov_loss_lambda,
        vocab_path=None,
        test_path=None,
    ) -> None:
        super(PointerGenerator, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            vocab_size=self.hparams.vocab_size + 4,  # special tokens
            embed_dim=self.hparams.embed_dim,
            hidden_dim=self.hparams.hidden_dim,
            num_layers=self.hparams.num_encoder_layer,
        )
        self.decoder = Decoder(
            vocab_size=self.hparams.vocab_size + 4,
            embed_dim=self.hparams.embed_dim,
            hidden_dim=self.hparams.hidden_dim,
            ptrgen_on=self.hparams.ptr_gen,
            coverage_on=self.hparams.coverage,
        )

    def forward(
        self,
        src: torch.tensor,
        tgt_input: torch.tensor,
        src_len: torch.tensor,
        src_padding_mask: torch.tensor,
        vocab: dict,
        src_oov: torch.tensor,
        content_selection=None,
    ):
        ## |src| = (B, T_s)
        ## |tgt_input| = (B, T_t)
        ## |src_len| = (B, )
        ## |src_padding_mask| = (B, T_s)
        ## |src_oov| = (B, T_s)
        ## |content_selection| = (B, T_s)
        temperature = 1  # do not use temperature in tranining phase

        encoder_output, hidden = self.encoder(src, src_len)

        B, T_s, D = encoder_output.shape
        context_vector = torch.zeros((B, D)).type_as(encoder_output)
        previous_coverage = (
            torch.zeros((B, T_s)).type_as(encoder_output) if self.hparams.coverage else None
        )
        if vocab is None:
            extra_zeros = None
        else:
            extra_vocab_size = len(vocab) - self.hparams.vocab_size
            extra_zeros = torch.zeros((B, extra_vocab_size)).type_as(encoder_output)

        outputs = []

        max_tgt_len = min(self.hparams.max_decoder_step, tgt_input.shape[1])
        for t in range(max_tgt_len):
            decoder_input = tgt_input[:, t]
            ## |decoder_input| = (B,)
            (
                vocab_dist_final,
                hidden,
                context_vector,
                attention_dist,
                next_coverage,
            ) = self.decoder(
                decoder_input,
                hidden,
                encoder_output,
                src_padding_mask,
                context_vector,
                t,
                extra_zeros,
                src_oov,
                previous_coverage,
                temperature,
                content_selection,
            )
            output = (vocab_dist_final,)
            ## |vocab_dist_final| = (B, V + oov)

            if self.hparams.coverage:
                cov_loss = torch.sum(torch.min(attention_dist, previous_coverage), 1)
                ## |cov_loss| = (B,)
                previous_coverage = next_coverage
                output += (cov_loss,)
            else:
                output += (None,)

            outputs.append(output)

        return outputs

    def training_step(self, batch: Tuple, batch_idx: int) -> dict:
        (
            src,
            tgt_input,
            tgt_output,
            src_len,
            tgt_len,
            src_padding_mask,
            tgt_padding_mask,
            src_oov,
            vocab,
        ) = batch
        outputs = self(src, tgt_input, src_len, src_padding_mask, vocab, src_oov)

        losses = []
        for t in range(len(outputs)):
            vocab_dist, cov_loss = outputs[t]
            decoder_target, target_mask = tgt_output[:, t], tgt_padding_mask[:, t]

            # Add 1e-12 to distribution for numerical stability
            loss = F.nll_loss(torch.log(vocab_dist + 1e-9), decoder_target, reduction="none")
            ## |loss| = (B,)

            if self.hparams.coverage:
                loss += self.hparams.cov_loss_lambda * cov_loss

            loss = loss * target_mask  # pad는 loss에서 제외
            losses.append(loss)

        avg_loss = torch.sum(torch.stack(losses, 1), 1) / tgt_len  # 길이별 평균
        loss = torch.mean(avg_loss)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> dict:
        (
            src,
            tgt_input,
            tgt_output,
            src_len,
            tgt_len,
            src_padding_mask,
            tgt_padding_mask,
            src_oov,
            vocab,
        ) = batch
        outputs = self(src, tgt_input, src_len, src_padding_mask, vocab, src_oov)

        losses = []
        for t in range(len(outputs)):
            vocab_dist, cov_loss = outputs[t]
            decoder_target, target_mask = tgt_output[:, t], tgt_padding_mask[:, t]

            loss = F.nll_loss(torch.log(vocab_dist + 1e-9), decoder_target, reduction="none")
            ## |loss| = (B,)

            if self.hparams.coverage:
                loss += self.hparams.cov_loss_lambda * cov_loss

            loss = loss * target_mask  # pad는 loss에서 제외
            losses.append(loss)

        avg_loss = torch.sum(torch.stack(losses, 1), 1) / tgt_len  # 길이별 평균
        loss = torch.mean(avg_loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs) -> dict:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)

    def test_step(self, batch, batch_idx) -> dict:
        pass

    def test_epoch_end(self, outputs) -> dict:
        pass

    def train_dataloader(self) -> DataLoader:
        self.train_dset = TextDataset(
            self.hparams.train_path, mode="train", use_sentencepiece=self.hparams.use_sentencepiece
        )
        collate_fn = Collator(
            dset=self.train_dset,
            ptr_gen=self.hparams.ptr_gen,
            max_len=self.hparams.max_len,
            decoder_max_len=self.hparams.max_decoder_step,
            use_sentencepiece=self.hparams.use_sentencepiece,
        )
        return DataLoader(
            self.train_dset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=16,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        self.val_dset = TextDataset(
            self.hparams.validation_path,
            mode="eval",
            vocab_dir=self.hparams.vocab_path,
            use_sentencepiece=self.hparams.use_sentencepiece,
        )
        collate_fn = Collator(
            dset=self.val_dset,
            ptr_gen=self.hparams.ptr_gen,
            max_len=self.hparams.max_len,
            decoder_max_len=self.hparams.max_decoder_step,
            use_sentencepiece=self.hparams.use_sentencepiece,
        )
        return DataLoader(
            self.val_dset,
            batch_size=self.hparams.eval_batch_size,
            collate_fn=collate_fn,
            num_workers=16,
        )

    def test_dataloader(self) -> DataLoader:
        self.test_dset = TextDataset(
            self.hparams.test_path,
            mode="test",
            vocab_dir=self.hparams.vocab_path,
            use_sentencepiece=self.hparams.use_sentencepiece,
        )
        collate_fn = Collator(
            dset=self.test_dset,
            ptr_gen=self.hparams.ptr_gen,
            max_len=self.hparams.max_len,
            decoder_max_len=self.hparams.max_decoder_step,
            use_sentencepiece=self.hparams.use_sentencepiece,
        )
        return DataLoader(self.test_dset, batch_size=5, shuffle=False, collate_fn=collate_fn)

    def configure_optimizers(self) -> optim:
        optimizer = optim.Adagrad(
            self.parameters(), lr=self.hparams.lr, initial_accumulator_value=0.1
        )
        # optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
