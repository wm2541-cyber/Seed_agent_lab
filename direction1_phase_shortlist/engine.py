"""Closed-loop Phase-Aware + Shortlist-Vocab speculative decoding engine on Qwen3.5-4B (MLX)."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
import numpy as np

from direction1_phase_shortlist.logz_calibrator import LogZCalibrator
from direction1_phase_shortlist.phase_router import (
    GenerationPhase,
    PhaseTagIds,
    RoutingConfig,
    infer_phase_from_tokens,
    policy_for_phase,
    step_phase,
)
from direction1_phase_shortlist.quantized_head import (
    DraftPrediction,
    MLXLogZHead,
    QuantizedSubHead,
    build_prompt_sub_head,
    compile_mlx_logz_head,
    predict_full_head_draft,
    predict_shortlist_draft,
    slice_quantized_embedding,
)
from direction1_phase_shortlist.shortlist_vocab import StaticShortlist
from seed_agent_lab.closed_loop_engine import (
    _VerifyGdnStash,
    _gdn_step_forward,
    _rollback_gdn_from_stash,
)
from seed_agent_lab.hybrid_speculative import find_prompt_lookup_draft


@dataclass(frozen=True)
class PhaseSpecResult:
    """Immutable telemetry and output from one closed-loop speculative generation run."""

    tokens: tuple[int, ...]
    elapsed_seconds: float
    tokens_per_sec: float
    verify_calls: int
    mean_accepted_tokens: float
    total_neural_steps: int
    total_lookup_steps: int
    rollback_count: int
    draft_seconds: float
    verify_seconds: float
    rollback_seconds: float
    sub_head_build_seconds: float
    effective_shortlist_size: int
    thinking_tokens: int
    response_tokens: int
    tool_call_tokens: int
    thinking_neural_steps: int
    non_thinking_neural_steps: int


def _truncate_at_first_eos(
    tokens: Sequence[int],
    eos_token_ids: frozenset[int],
) -> list[int]:
    """Purely truncate a draft sequence right after its first EOS token (inclusive)."""
    out: list[int] = []
    for tok in tokens:
        tok_i = int(tok)
        out.append(tok_i)
        if tok_i in eos_token_ids:
            break
    return out


class PhaseShortlistSpeculativeEngine:
    """Closed-loop greedy speculative decoder with Phase-Aware Routing and Shortlist + logZ calibration."""

    def __init__(
        self,
        model,
        w8_weight: np.ndarray,
        w8_bias: np.ndarray,
        phase_tags: PhaseTagIds,
        static_shortlist: StaticShortlist | None = None,
        logz_calibrator: LogZCalibrator | None = None,
        draft_layer: int = 8,
    ) -> None:
        self.model = model
        self.text_model = model.language_model.model
        self.lm = model.language_model
        self.draft_layer = draft_layer
        self.num_layers = len(model.layers)
        self.phase_tags = phase_tags

        self.w8_weight = mx.array(w8_weight, dtype=mx.bfloat16)
        self.w8_bias = mx.array(w8_bias, dtype=mx.bfloat16)
        mx.eval(self.w8_weight, self.w8_bias)

        self.draft_fa_idx = next((i for i in range(draft_layer) if not model.layers[i].is_linear), None)
        self.verify_fa_idx = next((i for i in range(draft_layer, self.num_layers) if not model.layers[i].is_linear), None)

        self.static_shortlist = static_shortlist
        self.static_token_set = static_shortlist.token_set if static_shortlist is not None else None
        self.static_sub_head: QuantizedSubHead | None = (
            slice_quantized_embedding(self.text_model.embed_tokens, static_shortlist.token_ids)
            if static_shortlist is not None
            else None
        )
        self.logz_head: MLXLogZHead | None = (
            compile_mlx_logz_head(logz_calibrator)
            if logz_calibrator is not None
            else None
        )

    def _full_lm_head(self, normed_hidden: mx.array) -> mx.array:
        if self.lm.args.tie_word_embeddings:
            return self.text_model.embed_tokens.as_linear(normed_hidden)
        return self.lm.lm_head(normed_hidden)

    def _draft_predict(
        self,
        hidden_at_k: mx.array,
        prompt_sub_head: QuantizedSubHead | None,
        use_shortlist: bool,
        use_logz_calibration: bool,
        logz_deadband_margin: float = 0.0,
    ) -> DraftPrediction:
        normed = self.text_model.norm(hidden_at_k)
        rotated = (normed @ self.w8_weight) + self.w8_bias
        if use_shortlist and prompt_sub_head is not None:
            active_cal = self.logz_head if use_logz_calibration else None
            return predict_shortlist_draft(
                rotated,
                prompt_sub_head,
                logz_head=active_cal,
                calibrator_hidden=normed,
                deadband_margin=logz_deadband_margin,
            )
        return predict_full_head_draft(rotated, self.text_model.embed_tokens)

    def _run_draft_layers_one_token(self, token_id: int, cache) -> mx.array:
        hidden = self.text_model.embed_tokens(mx.array([[token_id]], dtype=mx.int32))
        fa_mask = create_attention_mask(hidden, cache[self.draft_fa_idx]) if self.draft_fa_idx is not None else None
        for idx in range(self.draft_layer):
            layer = self.model.layers[idx]
            mask = None if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=cache[idx])
        return hidden

    def _run_verify_layers(
        self,
        hidden_chunk: mx.array,
        cache,
        start_layer: int,
        stash_gdn: bool,
    ) -> tuple[mx.array, list[_VerifyGdnStash]]:
        fa_idx = self.draft_fa_idx if start_layer == 0 else self.verify_fa_idx
        fa_mask = create_attention_mask(hidden_chunk, cache[fa_idx]) if fa_idx is not None else None
        stashes: list[_VerifyGdnStash] = []
        hidden = hidden_chunk
        for idx in range(start_layer, self.num_layers):
            layer = self.model.layers[idx]
            if layer.is_linear:
                hidden, st = _gdn_step_forward(layer, hidden, cache[idx], stash_out=stash_gdn, layer_idx=idx)
                if st is not None:
                    stashes.append(st)
            else:
                hidden = layer(hidden, mask=fa_mask, cache=cache[idx])
        logits = self._full_lm_head(self.text_model.norm(hidden))[0]
        return logits, stashes

    def generate_baseline(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_ids: Sequence[int] | set[int] | frozenset[int],
    ) -> tuple[tuple[int, ...], float]:
        """Standard full 32-layer greedy baseline on the exact same model and cache."""
        eos_set = frozenset(int(t) for t in eos_token_ids)
        cache = self.model.make_cache()
        logits = self.model(mx.array([list(prompt_ids)], dtype=mx.int32), cache=cache)[0, -1, :]
        next_tok = mx.argmax(logits, axis=-1)
        mx.eval(next_tok)
        cur_tok = int(next_tok.item())

        generated: list[int] = []
        t0 = time.perf_counter()
        while len(generated) < max_new_tokens:
            generated.append(cur_tok)
            if cur_tok in eos_set or len(generated) >= max_new_tokens:
                break
            logits = self.model(mx.array([[cur_tok]], dtype=mx.int32), cache=cache)[0, -1, :]
            next_tok = mx.argmax(logits, axis=-1)
            mx.eval(next_tok)
            cur_tok = int(next_tok.item())

        elapsed = time.perf_counter() - t0
        del cache
        mx.clear_cache()
        return tuple(generated), elapsed

    def generate_speculative(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_ids: Sequence[int] | set[int] | frozenset[int],
        routing: RoutingConfig,
        use_shortlist: bool = True,
        use_logz_calibration: bool = True,
        logz_deadband_margin: float = 0.0,
        stop_draft_on_eos: bool = True,
    ) -> PhaseSpecResult:
        """True closed-loop speculative decoding matching greedy baseline token-for-token."""
        eos_set = frozenset(int(t) for t in eos_token_ids)
        full_vocab_size = int(self.text_model.embed_tokens.weight.shape[0])

        t_sub0 = time.perf_counter()
        prompt_sub_head: QuantizedSubHead | None = None
        effective_shortlist_size = full_vocab_size
        if use_shortlist and self.static_shortlist is not None and self.static_sub_head is not None:
            prompt_sub_head, dynamic_sl = build_prompt_sub_head(
                self.text_model.embed_tokens,
                self.static_shortlist,
                self.static_sub_head,
                prompt_ids,
                static_token_set=self.static_token_set,
            )
            effective_shortlist_size = dynamic_sl.total_size
        sub_head_build_seconds = time.perf_counter() - t_sub0

        cache = self.model.make_cache()
        logits = self.model(mx.array([list(prompt_ids)], dtype=mx.int32), cache=cache)[0, -1, :]
        next_tok = mx.argmax(logits, axis=-1)
        mx.eval(next_tok)
        cur_tok = int(next_tok.item())

        history_buf = [int(t) for t in prompt_ids]
        generated: list[int] = []
        phase = infer_phase_from_tokens(prompt_ids, self.phase_tags)

        verify_calls = 0
        total_neural_steps = 0
        total_lookup_steps = 0
        rollback_count = 0
        draft_seconds = 0.0
        verify_seconds = 0.0
        rollback_seconds = 0.0
        thinking_neural_steps = 0
        non_thinking_neural_steps = 0
        phase_counts: dict[GenerationPhase, int] = {
            GenerationPhase.THINKING: 0,
            GenerationPhase.RESPONSE: 0,
            GenerationPhase.TOOL_CALL: 0,
        }

        def _build_result(elapsed: float) -> PhaseSpecResult:
            return PhaseSpecResult(
                tokens=tuple(generated),
                elapsed_seconds=elapsed,
                tokens_per_sec=len(generated) / elapsed if elapsed > 0 else 0.0,
                verify_calls=verify_calls,
                mean_accepted_tokens=len(generated) / verify_calls if verify_calls > 0 else 0.0,
                total_neural_steps=total_neural_steps,
                total_lookup_steps=total_lookup_steps,
                rollback_count=rollback_count,
                draft_seconds=draft_seconds,
                verify_seconds=verify_seconds,
                rollback_seconds=rollback_seconds,
                sub_head_build_seconds=sub_head_build_seconds,
                effective_shortlist_size=effective_shortlist_size,
                thinking_tokens=phase_counts[GenerationPhase.THINKING],
                response_tokens=phase_counts[GenerationPhase.RESPONSE],
                tool_call_tokens=phase_counts[GenerationPhase.TOOL_CALL],
                thinking_neural_steps=thinking_neural_steps,
                non_thinking_neural_steps=non_thinking_neural_steps,
            )

        t_start = time.perf_counter()
        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            if cur_tok in eos_set or remaining == 1:
                phase_counts[phase] += 1
                generated.append(cur_tok)
                break

            phase_counts[phase] += 1
            generated.append(cur_tok)
            history_buf.append(cur_tok)
            phase = step_phase(phase, cur_tok, self.phase_tags)
            remaining -= 1

            policy = policy_for_phase(routing, phase)

            td0 = time.perf_counter()
            draft_tokens: list[int] = []
            lookup_steps = 0
            neural_steps = 0

            if policy.use_lookup and policy.lookup_first:
                copied = find_prompt_lookup_draft(
                    history_buf,
                    ngram_size=policy.ngram_size,
                    max_draft_len=min(policy.max_lookup_len, remaining),
                )
                if copied:
                    filtered_copy = _truncate_at_first_eos(copied, eos_set) if stop_draft_on_eos else list(copied)
                    draft_tokens.extend(filtered_copy)
                    lookup_steps = len(filtered_copy)

            draft_gdn_snapshots: list[list[tuple[mx.array | None, mx.array | None]]] = []
            draft_hiddens: list[mx.array] = []

            if policy.use_neural and lookup_steps == 0 and policy.max_neural_len > 0:
                feed_tok = cur_tok
                draft_phase = phase
                limit = min(policy.max_neural_len, remaining)
                while neural_steps < limit:
                    h8 = self._run_draft_layers_one_token(feed_tok, cache)
                    draft_hiddens.append(h8)
                    snap = [
                        (cache[i][0], cache[i][1])
                        for i in range(self.draft_layer)
                        if self.model.layers[i].is_linear
                    ]
                    draft_gdn_snapshots.append(snap)

                    pred = self._draft_predict(
                        h8,
                        prompt_sub_head=prompt_sub_head,
                        use_shortlist=use_shortlist,
                        use_logz_calibration=use_logz_calibration,
                        logz_deadband_margin=logz_deadband_margin,
                    )
                    draft_tokens.append(pred.token_id)
                    neural_steps += 1
                    if draft_phase is GenerationPhase.THINKING:
                        thinking_neural_steps += 1
                    else:
                        non_thinking_neural_steps += 1
                    feed_tok = pred.token_id

                    if stop_draft_on_eos and pred.token_id in eos_set:
                        break
                    if policy.confidence_threshold > 0.0 and pred.calibrated_prob < policy.confidence_threshold:
                        break
                    next_draft_phase = step_phase(draft_phase, pred.token_id, self.phase_tags)
                    if not policy_for_phase(routing, next_draft_phase).use_neural:
                        break
                    draft_phase = next_draft_phase
            draft_seconds += time.perf_counter() - td0

            tv0 = time.perf_counter()
            verify_calls += 1
            total_neural_steps += neural_steps
            total_lookup_steps += lookup_steps

            if neural_steps > 0:
                h8_chunk = mx.concatenate(draft_hiddens, axis=1)
                k_total = neural_steps
                verify_logits, gdn_stashes = self._run_verify_layers(
                    h8_chunk, cache, start_layer=self.draft_layer, stash_gdn=(k_total > 1)
                )
            else:
                verify_inputs = [cur_tok] + draft_tokens[:-1]
                k_total = len(verify_inputs)
                x_emb = self.text_model.embed_tokens(mx.array([verify_inputs], dtype=mx.int32))
                verify_logits, gdn_stashes = self._run_verify_layers(
                    x_emb, cache, start_layer=0, stash_gdn=(k_total > 1)
                )

            target_toks_mx = mx.argmax(verify_logits, axis=-1)
            mx.eval(target_toks_mx)
            target_toks = [int(x) for x in target_toks_mx.tolist()]
            verify_seconds += time.perf_counter() - tv0

            matched = 0
            for d_tok, t_tok in zip(draft_tokens, target_toks):
                if d_tok == t_tok:
                    matched += 1
                else:
                    break

            n_keep = min(matched + 1, k_total)
            if n_keep < k_total:
                tr0 = time.perf_counter()
                rollback_count += 1
                drop_kv = k_total - n_keep

                for i in range(self.num_layers):
                    if not self.model.layers[i].is_linear:
                        cache[i].trim(drop_kv)

                if neural_steps > 0:
                    lin_idx = 0
                    for i in range(self.draft_layer):
                        if self.model.layers[i].is_linear:
                            cache[i][0], cache[i][1] = draft_gdn_snapshots[n_keep - 1][lin_idx]
                            lin_idx += 1

                for st in gdn_stashes:
                    _rollback_gdn_from_stash(self.model.layers[st.layer_idx], cache[st.layer_idx], st, n_keep)

                mx.eval([c[1] for layer, c in zip(self.model.layers, cache) if layer.is_linear and c[1] is not None])
                rollback_seconds += time.perf_counter() - tr0

            if matched > 0:
                all_matched = matched == len(draft_tokens)
                commit_count = matched - 1 if all_matched else matched
                for idx in range(matched):
                    tok = draft_tokens[idx]
                    if tok in eos_set:
                        phase_counts[phase] += 1
                        generated.append(tok)
                        elapsed = time.perf_counter() - t_start
                        del cache
                        mx.clear_cache()
                        return _build_result(elapsed)
                    if idx < commit_count:
                        phase_counts[phase] += 1
                        generated.append(tok)
                        history_buf.append(tok)
                        phase = step_phase(phase, tok, self.phase_tags)
                cur_tok = draft_tokens[-1] if all_matched else target_toks[matched]
            else:
                cur_tok = target_toks[0]

        elapsed = time.perf_counter() - t_start
        del cache
        mx.clear_cache()
        return _build_result(elapsed)
