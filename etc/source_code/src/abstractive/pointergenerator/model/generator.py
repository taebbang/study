import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[UNK]"
START_DECODING = "[START]"
STOP_DECODING = "[STOP]"

special_tokens = [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]


class SentenceDecoder:
    def __init__(self, pl_model):
        self.model = pl_model
        self.hparams = pl_model.hparams

    def generate(
        self,
        src: torch.tensor,
        src_len: torch.tensor,
        src_padding_mask: torch.tensor,
        vocab: dict,
        src_oov: torch.tensor,
        min_length: int = 1,
        num_beams: int = 4,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        early_stopping: bool = False,
        temperature: float = 1,
        top_k: int = -1,
        top_p: float = -1.0,
        length_penalty: float = 0.0,
        copy_penalty_beta: float = 0.0,
        no_repeat_ngram_size: int = 0,
        contents: torch.tensor = None,
        contents_threshold: float = 0.2,
    ):
        """Adapted from huggingface transformers"""
        ## |src| = (B, T_s)
        ## |src_len| = (B, )
        ## |src_padding_mask| = (B, T_s)
        ## |src_oov| = (B, T_s)

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        self.model.eval()
        torch.set_grad_enabled(False)
        B, T_s = src.shape
        encoder_output, hidden = self.model.encoder(src, src_len)
        ## |encoder_output|, |hidden| = (B, T_s, D), ((1, B, D/2), (1, B, D/2))

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = B * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = B
            effective_batch_mult = 1

        if contents is not None:
            # mask contents lower than threshold
            contents = (contents > contents_threshold).float()

        # Expand src if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            src, src_padding_mask, src_oov = [
                s.unsqueeze(1)
                .expand(B, effective_batch_mult * num_beams, T_s)
                .contiguous()
                .view(effective_batch_size * num_beams, T_s)
                for s in (src, src_padding_mask, src_oov)
            ]
            ## |src| = (effective_batch_size * num_beams, T_s)
            ## |src_padding_mask| = (effective_batch_size * num_beams, T_s)
            ## |src_oov| = (effective_batch_size * num_beams, T_s)
            if contents is not None:
                contents = (
                    contents.unsqueeze(1)
                    .expand(B, effective_batch_mult * num_beams, T_s)
                    .contiguous()
                    .view(effective_batch_size * num_beams, T_s)
                )

            src_len = (
                src_len.unsqueeze(1)
                .expand(B, effective_batch_mult * num_beams)
                .contiguous()
                .view(effective_batch_size * num_beams)
            )

            expanded_batch_idxs = (
                torch.arange(B)
                .view(-1, 1)
                .repeat(1, effective_batch_mult * num_beams)
                .view(-1)
                .type_as(src_len)
            )

            # expand encoder outputs and hidden
            encoder_output = encoder_output.index_select(0, expanded_batch_idxs)
            hidden = tuple(h.index_select(1, expanded_batch_idxs) for h in hidden)
            ## |encoder_output| = (effective_batch_size * num_beams, T_s, D)

        D = encoder_output.shape[-1]

        context_vector = torch.zeros((effective_batch_size * num_beams, D)).type_as(encoder_output)
        previous_coverage = (
            torch.zeros((effective_batch_size * num_beams, T_s)).type_as(encoder_output)
            if self.hparams.coverage
            else None
        )
        if src_oov is None:
            extra_zeros = None
        else:
            extra_vocab_size = len(vocab) - self.hparams.vocab_size + 4
            extra_zeros = torch.zeros((effective_batch_size * num_beams, extra_vocab_size)).type_as(
                encoder_output
            )

        tgt_input = (
            torch.zeros((effective_batch_size * num_beams, 1))
            .fill_(vocab[START_DECODING])
            .type_as(src)
        )

        if num_beams > 1:
            output = self._generate_beam_search(
                tgt_input,
                encoder_output=encoder_output,
                hidden=hidden,
                src_padding_mask=src_padding_mask,
                context_vector=context_vector,
                extra_zeros=extra_zeros,
                src_oov=src_oov,
                coverage=previous_coverage,
                min_length=min_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                batch_size=B,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                vocab=vocab,
                length_penalty=length_penalty,
                copy_penalty_beta=copy_penalty_beta,
                src_len=src_len,
                no_repeat_ngram_size=no_repeat_ngram_size,
                contents=contents,
            )
        else:
            output = self._generate_no_beam_search(
                tgt_input,
                encoder_output=encoder_output,
                hidden=hidden,
                src_padding_mask=src_padding_mask,
                context_vector=context_vector,
                extra_zeros=extra_zeros,
                src_oov=src_oov,
                coverage=previous_coverage,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                vocab=vocab,
                no_repeat_ngram_size=no_repeat_ngram_size,
                contents=contents,
            )
        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        encoder_output,
        hidden,
        src_padding_mask,
        context_vector,
        extra_zeros,
        src_oov,
        coverage,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        vocab,
        no_repeat_ngram_size,
        contents,
    ):
        cur_len = 1

        B = encoder_output.shape[0]

        # length of generated / unfinished sentences
        sent_lengths = input_ids.new(B).fill_(self.hparams.max_decoder_step)
        unfinished_sents = input_ids.new(B).fill_(1)
        total_attention = []
        while cur_len < self.hparams.max_decoder_step:
            decoder_input = input_ids[:, -1]
            is_unk = decoder_input >= self.hparams.vocab_size + 4  # consider special tokens
            decoder_input.masked_fill_(is_unk, vocab[UNKNOWN_TOKEN])
            (vocab_dist, hidden, context_vector, attention_dist, coverage,) = self.model.decoder(
                decoder_input,
                hidden,
                encoder_output,
                src_padding_mask,
                context_vector,
                cur_len,
                extra_zeros,
                src_oov,
                coverage,
                temperature,
                contents,
            )
            ## |vocab_dist| = (B, V + oov)
            _scores = torch.log(vocab_dist + 1e-9)
            _scores = postprocess_next_token_scores(
                scores=_scores,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                cur_len=cur_len,
                min_length=min_length,
                max_length=self.hparams.max_decoder_step,
                eos_token_id=vocab[STOP_DECODING],
                batch_size=B,
                num_beams=1,
            )
            vocab_dist = torch.exp(_scores)
            # set eos token prob to zero if min_length is not reached
            if cur_len < min_length:
                vocab_dist[:, vocab[STOP_DECODING]] = 0

            if do_sample:
                # Sampling
                vocab_dist = top_k_top_p_filtering(vocab_dist, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(vocab_dist, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(vocab_dist, dim=-1)

            # pad finished sentences
            tokens_to_add = next_token * unfinished_sents + (vocab[PAD_TOKEN]) * (
                1 - unfinished_sents
            )

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len += 1

            eos_in_sents = tokens_to_add == vocab[STOP_DECODING]
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(
                eos_in_sents.long()
            ).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        return input_ids

    def _generate_beam_search(
        self,
        input_ids,
        encoder_output,
        hidden,
        src_padding_mask,
        context_vector,
        extra_zeros,
        src_oov,
        coverage,
        min_length,
        num_beams,
        num_return_sequences,
        batch_size,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        vocab,
        length_penalty,
        copy_penalty_beta,
        src_len,
        no_repeat_ngram_size,
        contents,
    ):
        cur_len = 1

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, self.hparams.max_decoder_step, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams)).type_as(context_vector)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # done sentences
        done = [False for _ in range(batch_size)]
        total_attention = torch.zeros_like(src_oov).type_as(context_vector)
        while cur_len < self.hparams.max_decoder_step:
            decoder_input = input_ids[:, -1]
            is_unk = decoder_input >= self.hparams.vocab_size + 4
            decoder_input.masked_fill_(is_unk, vocab[UNKNOWN_TOKEN])
            (vocab_dist, hidden, context_vector, attention_dist, coverage,) = self.model.decoder(
                decoder_input,
                hidden,
                encoder_output,
                src_padding_mask,
                context_vector,
                cur_len,
                extra_zeros,
                src_oov,
                coverage,
                temperature,
                contents,
            )
            ## |total_attention| = (B * num_beams, T_s)
            total_attention += attention_dist

            ## |vocab_dist| = (B * num_beams, V + oov)
            vocab_size = vocab_dist.shape[-1]

            # Take log to dist to sum whole probabiltiies (Add 1e-12 for numerical stability)
            _scores = torch.log(vocab_dist + 1e-9)
            _scores = postprocess_next_token_scores(
                scores=_scores,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                cur_len=cur_len,
                min_length=min_length,
                max_length=self.hparams.max_decoder_step,
                eos_token_id=vocab[STOP_DECODING],
                batch_size=batch_size,
                num_beams=num_beams,
                length_penalty=length_penalty,
                copy_penalty_beta=copy_penalty_beta,
                total_attention=total_attention,
                src_len=src_len,
            )
            # set eos token prob to zero if min_length is not reached
            if cur_len < min_length:
                _scores[:, vocab[STOP_DECODING]] = -float("inf")

            if do_sample:
                _scores = _scores + beam_scores[:, None].expand_as(_scores)
                _scores = top_k_top_p_filtering(_scores, top_k=top_k, top_p=top_p)
                ## |_scores| = (B * num_beams, V + oov)

                # re-organize to group the beam togehter to sample from all beam_idxs
                _scores = _scores.contiguous().view(batch_size, num_beams * vocab_size)
                ## |_scores| = (B, num_beam * (V + oov))

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                vocab_dist = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(vocab_dist, num_samples=num_beams * 2)
                ## |next_tokens| = (B, num_beams * 2)

                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)
                ## |next_scores| = (B, num_beams * 2)

                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)
                ## |next_tokens| = (B, num_beams * 2)  # sorted
            else:
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(batch_size, num_beams * vocab_size)
                next_scores, next_tokens = torch.topk(
                    next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)
            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):
                # if we are done with this sentence, add a pad token
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(
                        num_beams
                    )
                    next_batch_beam.extend([(0, vocab[PAD_TOKEN], 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence
                    if token_id.item() == vocab[STOP_DECODING]:
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # once the beam for next step is full, don't add more tokens to it.
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (
                    batch_idx + 1
                ), "We should have added num_beams each step"

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if all(
                (token_id % vocab_size).item() != vocab[STOP_DECODING]
                for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams]
                    == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx],
                    beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.hparams.max_decoder_step)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(vocab[PAD_TOKEN])

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.hparams.max_decoder_step:
                    decoded[i, sent_lengths[i]] = vocab[STOP_DECODING]
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == self.hparams.max_decoder_step for hypo in best)
            decoded = torch.stack(best).type_as(input_ids)

        return decoded


def top_k_top_p_filtering(
    prob: torch.tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        prob: vocab distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), prob.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = prob < torch.topk(prob, top_k)[0][..., -1, None]
        prob[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_prob, sorted_indices = torch.sort(prob, descending=True)
        cumulative_probs = torch.cumsum(sorted_prob, dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        prob[indices_to_remove] = filter_value
    return prob


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, early_stopping, length_penalty=0):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def postprocess_next_token_scores(
    scores,
    input_ids,
    no_repeat_ngram_size,
    cur_len,
    min_length,
    max_length,
    eos_token_id,
    batch_size,
    num_beams,
    length_penalty=0,
    copy_penalty_beta=0,
    total_attention=None,
    src_len=None,
):

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    def length_normalization(cur_len):
        # length penalty from Google's Neural Machine Translation System.
        return ((5 + cur_len) ** length_penalty) / (5 + 1) ** length_penalty

    scores = scores / length_normalization(cur_len)

    if copy_penalty_beta > 0:
        attention = total_attention.where(total_attention > 1, torch.ones_like(total_attention))
        len_mask = pad_sequence([torch.ones(i) for i in src_len], batch_first=True).type_as(
            attention
        )
        attention = attention * len_mask
        copy_penalty = copy_penalty_beta * (attention.sum(-1) - src_len)
        scores = scores - copy_penalty.unsqueeze(-1)
    return scores


def calc_banned_ngram_tokens(
    prev_input_ids: torch.tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int
) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [
                ngram[-1]
            ]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def set_scores_to_inf_for_banned_tokens(scores, banned_tokens) -> None:
    """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
    # [ 0  1  1 ]
    # [ 0  0  0 ]
    # [ 1  0  0 ]

    banned_mask = (
        torch.sparse.LongTensor(banned_mask.t(), indices, scores.size())
        .to(scores.device)
        .to_dense()
        .bool()
    )
    scores.masked_fill_(banned_mask, -float("inf"))
