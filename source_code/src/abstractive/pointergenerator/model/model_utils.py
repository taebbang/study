import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Dict, Tuple
import copy


class Encoder(nn.Module):
    """Single layer bidirectional LSTM.

    :param input_dim: vocab size(input to the encoder).
    :param embed_dim: the dimension of embedding layer.
    :param hidden_dim: the dimension of the hidden and cell states.

    :dim B: batch size
    :dim T: sequence length of each time step
    :dim E: word embedding dimension
    :dim D: hidden dimension
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = True,
    ) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_dim
        self.num_direction = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim)
        ## TODO: weight initialization

    def forward(self, _input: torch.tensor, seq_lens: torch.tensor) -> Tuple[torch.tensor]:
        ## |_input| = (B, T)
        ## |seq_lens| = (B, )

        embedded = self.embedding(_input)
        ## |embedded| = (B, T, E)
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True, enforce_sorted=False)

        self.rnn.flatten_parameters()  # for MultiGPU training
        output, (hidden, cell) = self.rnn(packed)
        ## |output| = (B, T, D*2)
        ## |hidden|, |cell| = (layer*direction, B, D), (layer*direction, B, D)

        # combine bidirectional(forward/backward) lstm
        B = _input.shape[0]
        hidden = hidden.view(1, B, self.hidden_size * self.num_direction)
        cell = cell.view(1, B, self.hidden_size * self.num_direction)
        ## |hidden|, |cell| = (1, B, D*2), (1, B, D*2)

        hidden = self.reduce_h(hidden)
        cell = self.reduce_c(cell)
        ## |hidden|, |cell| = (1, B, D), (1, B, D)

        output, _ = pad_packed_sequence(output, batch_first=True)
        ## |output| = (B, T, D*2)
        return output, (hidden, cell)


class Attention(nn.Module):
    """Bahdanau et al, 2015

    :param encdec_hidden: hidden size of both encoder and decoder(it does not need to be same).

    :dim B: batch size
    :dim T: sequence length of each time step
    :dim E: word embedding dimension
    :dim D: hidden dimension (both encoder and decoder)
    """

    def __init__(self, hidden_dim: int, coverage_on: bool = False) -> None:
        super(Attention, self).__init__()
        self.coverage_on = coverage_on
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.W_c = nn.Linear(1, hidden_dim * 2, bias=False)
        self.W_s = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.tensor,
        encoder_outputs: torch.tensor,
        encoder_pad_mask: torch.tensor,
        coverage: torch.tensor = None,
        softmax_temperature: float = 1,
    ) -> torch.tensor:
        ## |decoder_hidden| = (1, B, D*2)
        ## |encoder_outputs| = (B, T, D*2)
        ## |encoder_pad_mask| = (B, T)
        ## |coverage| = (B, T)

        src_len = encoder_outputs.shape[1]

        encoder_feature = self.W_h(encoder_outputs)
        ## |encoder_feature| = (B, T, D*2)

        decoder_hidden_expanded = decoder_hidden.transpose(0, 1).repeat(1, src_len, 1)
        ## |hidden| = (B, T, D*2) # Repeat the same vectors to time axis
        decoder_feature_expanded = self.W_s(decoder_hidden_expanded)
        ## |decoder_feature| = (B, T, D*2)

        if self.coverage_on:
            coverage_featrue = self.W_c(coverage.unsqueeze(-1))
            ## |coverage_feature| = (B, T, D*2)
            decoder_feature_expanded = decoder_feature_expanded + coverage_featrue

        energy = torch.tanh(encoder_feature + decoder_feature_expanded)
        ## |energy| = (B, T, D*2)

        scores = self.v(energy).squeeze(-1)
        ## |scores| = (B, T)

        attention_dist = (
            F.softmax(scores / softmax_temperature, dim=-1) * encoder_pad_mask
        )  # Do not attend to PAD
        attention_dist = attention_dist / attention_dist.sum(-1, keepdim=True)
        ## |attention_dist| = (B, T)

        context_vector = torch.bmm(attention_dist.unsqueeze(1), encoder_outputs).squeeze(1)
        ## |context_vector| = (B, D*2)

        if self.coverage_on:
            coverage = coverage.squeeze(-1) + attention_dist

        return context_vector, attention_dist, coverage


class Decoder(nn.Module):
    """Single layer unidirectional LSTM.

    :param output_dim: vocab size.
    :param embed_dim: the dimension of embedding layer.
    :param hidden_dim: the dimension of the hidden and cell states.
    :param attention: attention mechanism(Bahdanau et al, 2015).

    :dim B: batch size
    :dim T: sequence length of each time step
    :dim E: word embedding dimension
    :dim D: hidden dimension
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        ptrgen_on: bool,
        coverage_on: bool,
        num_layers: int = 1,
    ) -> None:
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.ptrgen_on = ptrgen_on

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.generate_x = nn.Linear(hidden_dim * 2 + embed_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True,)
        self.attention_network = Attention(hidden_dim, coverage_on)
        self.ptrgen_linear = nn.Linear(hidden_dim * 4 + embed_dim, 1)

        self.V1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.V2 = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        decoder_input: torch.tensor,
        decoder_hidden: Tuple[torch.tensor],
        encoder_outputs: torch.tensor,
        encoder_pad_mask: torch.tensor,
        previous_context_vector: torch.tensor,
        step: int,
        extra_zeros: torch.tensor = None,
        encoder_input_extra_vocabs: torch.tensor = None,
        coverage: torch.tensor = None,
        softmax_temperature: float = 1,
        content_selection=None,
    ) -> Tuple[torch.tensor]:
        ## |decoder_input| = (B,)
        ## |decoder_hidden| = ((1, B, D), (1, B, D))
        ## |encoder_outputs| = (B, T, D*2)
        ## |encoder_pad_mask| = (B, T)
        ## |previous_context_vector| = (B, D*2)
        ## |extra_zeros| = (B, oov_size)
        ## |encoder_input_extra_vocabs| = (B, T)
        ## |coverage| = (B, T) or None
        ## |content_selection| = (B, T) or None

        if not self.training and step == 0:
            # 처음 inference 하는 경우 이전 context vector와 coverage가 모두 0으로 이루어짐
            # 첫 예측 시 coverage를 생성하지 않기 위해 이렇게 설계
            decoder_hat = torch.cat(decoder_hidden, -1)
            ## |decoder_hat| = (1, B, D*2)
            _, _, coverage = self.attention_network(
                decoder_hat, encoder_outputs, encoder_pad_mask, coverage,
            )
            ## |context_vector| = (B, D*2)

        embedded = self.embedding(decoder_input).unsqueeze(1)
        ## |embedded| = (B, 1, E)

        input_vector = self.generate_x(
            torch.cat((previous_context_vector.unsqueeze(1), embedded), -1)
        )
        ## |input_vector| = (B, 1, E)

        self.rnn.flatten_parameters()  # for MultiGPU training
        output, decoder_hidden_next = self.rnn(input_vector, decoder_hidden)
        ## |output| = (B, 1, D)
        ## |decoder_hidden_next| = ((1, B, D), (1, B, D))

        decoder_hat = torch.cat(decoder_hidden_next, -1)
        ## |decoder_hat| = (1, B, D*2)

        context_vector, attention_dist, coverage_next = self.attention_network(
            decoder_hat, encoder_outputs, encoder_pad_mask, coverage, softmax_temperature,
        )
        ## |context_vector| = (B, D*2)
        ## |attention_dist, coverage_next| = (B, T), (B, T) or None

        if content_selection is not None:
            attention_dist_ = attention_dist * content_selection
            # attention_dist /= attention_dist.sum(-1).unsqueeze(1)
            # set a normalization parameter to 2 in bottom up attention paper
            attention_dist_ *= 2
        else:
            attention_dist_ = attention_dist

        if self.training or step > 0:
            coverage = coverage_next

        if self.ptrgen_on:
            ptrgen_input = torch.cat(
                (context_vector, decoder_hat.squeeze(0), input_vector.squeeze(1)), -1
            )
            ## |ptrgen_input| = (B, D*4 + E)
            p_gen = torch.sigmoid(self.ptrgen_linear(ptrgen_input))
            ## |p_gen| = (B, 1)

        output = torch.cat((output.squeeze(1), context_vector), 1)
        ## |output| = (B, D*3)

        output = self.V2(self.V1(output))
        ## |output| = (B, V)
        output /= softmax_temperature

        vocab_dist = F.softmax(output, dim=-1)

        if self.ptrgen_on:
            vocab_dist_oov = p_gen * vocab_dist
            attention_dist_oov = (1 - p_gen) * attention_dist_
            # |vocab_dist_oov| = (B, V)
            # |attention_dist_oov| = (B, T)

            if extra_zeros is not None:
                # batch 내부에 out-of-vocabulary가 존재하는 경우
                vocab_dist_oov = torch.cat((vocab_dist_oov, extra_zeros), 1)  # 새로 등장하는 단어만큼 차원 증가
                ## |vocab_dist_oov| = (B, V + oov_size)

            vocab_dist_final = vocab_dist_oov.scatter_add(
                1, encoder_input_extra_vocabs, attention_dist_oov
            )  # encoder_input_extra_vocabs의 값을 attention_dist_oov의 index로 사용해 해당 위치의 value를 vocab_dist에 더한다.
        else:
            vocab_dist_final = vocab_dist
            ## |vocab_dist| = (B, V)

        return (
            vocab_dist_final,
            decoder_hidden_next,
            context_vector,
            attention_dist,
            coverage,
        )


class Collator:
    def __init__(
        self,
        dset: torch.utils.data.Dataset,
        ptr_gen: bool,
        max_len: int,
        decoder_max_len: int,
        use_sentencepiece: bool = False,
    ) -> None:
        self.dataset = dset
        self.do_ptr_gen = ptr_gen
        self.max_len = max_len
        self.decoder_max_len = decoder_max_len
        self.use_sentencepiece = use_sentencepiece

    def pad(
        self, sample: List[int], max_len: int, pad_idx: int, preserve_last: bool = False
    ) -> List[int]:
        diff = max_len - len(sample)
        if diff > 0:
            return sample + [pad_idx] * diff
        else:
            if preserve_last:
                # Preserve last token([STOP]) of target output
                return sample[: max_len - 1] + [sample[-1]]
            return sample[:max_len]

    def __call__(self, batch: Tuple[list]) -> Tuple[torch.LongTensor]:
        src, tgt, src_oov_vocabs, tgt_oov_vocabs = zip(*batch)

        src_len = torch.tensor([self.max_len if len(s) > self.max_len else len(s) for s in src])
        tgt_len = torch.tensor(
            [self.decoder_max_len if len(s) > self.decoder_max_len else len(s) for s in tgt]
        )

        src_max_len, tgt_max_len = max(src_len).item(), max(tgt_len).item()

        # 매 배치마다 vocab_with_oov 생성 -> 이걸 통해 oov 단어도 generation 가능
        if self.do_ptr_gen and not self.use_sentencepiece:
            oov_list = list(set(x for sublist in src_oov_vocabs for x in sublist.keys()))
            vocab_with_oov = copy.deepcopy(self.dataset.vocab)
            for i in range(len(oov_list)):
                vocab_with_oov[oov_list[i]] = len(self.dataset.vocab) + i

            src_oov, tgt_oov = [copy.deepcopy(i) for i in (src, tgt)]

            # 3중 반복문이지만 문장마다 oov가 많지 않기 때문에 실제 계산량은 별로 없을듯..? -> 체크해보기
            for i, oovs in enumerate(src_oov_vocabs):  # n: batch size
                for oov in oovs.keys():  # n: the number of OOV words in article
                    for idx in oovs[oov]:  # n : the number of appearance of the OOV word
                        src_oov[i][idx] = vocab_with_oov[oov]
                        # 결국 batch size 안의 모든 OOV word에 대해 반복문을 돌린 것과 동일(O(N))

            for i, oovs in enumerate(tgt_oov_vocabs):
                for oov in oovs.keys():
                    for idx in oovs[oov]:
                        tgt_oov[i][idx] = vocab_with_oov[oov]

        pad_idx = self.dataset.vocab["[PAD]"]

        src = torch.LongTensor([self.pad(s, src_max_len, pad_idx) for s in src])
        tgt_input = torch.LongTensor([self.pad(t[:-1], tgt_max_len, pad_idx) for t in tgt])

        src_padding_mask = src != pad_idx

        if self.do_ptr_gen and not self.use_sentencepiece:
            src_oov = torch.LongTensor([self.pad(s, src_max_len, pad_idx) for s in src_oov])
            tgt_output_oov = torch.LongTensor(
                [self.pad(t[1:], tgt_max_len, pad_idx, True) for t in tgt_oov]
            )
            tgt_padding_mask = tgt_output_oov != pad_idx
            return (
                src,
                tgt_input,
                tgt_output_oov,
                src_len,
                tgt_len,
                src_padding_mask,
                tgt_padding_mask,
                src_oov,
                vocab_with_oov,
            )
        else:
            tgt_output = torch.LongTensor(
                [self.pad(t[1:], tgt_max_len, pad_idx, True) for t in tgt]
            )
            tgt_padding_mask = tgt_output != pad_idx
            return (
                src,
                tgt_input,
                tgt_output,
                src_len,
                tgt_len,
                src_padding_mask,
                tgt_padding_mask,
                src,
                None,
            )
