"""Unified End-to-End Inference Pipeline combining Direction 2/3 Prefill with Direction 1 Speculative Decode."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import mlx.core as mx
import numpy as np

from direction1_phase_shortlist.engine import PhaseShortlistSpeculativeEngine, _truncate_at_first_eos
from direction1_phase_shortlist.phase_router import (
    GenerationPhase,
    PhaseTagIds,
    RoutingConfig,
    infer_phase_from_tokens,
    phase_gated_routing,
    policy_for_phase,
    step_phase,
)
from direction1_phase_shortlist.quantized_head import QuantizedSubHead, build_prompt_sub_head
from direction1_phase_shortlist.shortlist_vocab import StaticShortlist
from direction2_hybrid_prefill_align.aligned_prefill import (
    AlignedPrefillConfig,
    CompiledHybridAligner,
    run_aligned_shallow_prefill,
)
from direction3_role_prefill_svd.role_segments import build_depth_schedule, parse_role_tags
from seed_agent_lab.closed_loop_engine import _rollback_gdn_from_stash
from seed_agent_lab.hybrid_speculative import find_prompt_lookup_draft


@dataclass(frozen=True)
class UnifiedPipelineConfig:
    """Immutable end-to-end configuration pairing a Prefill policy with a Decode policy."""

    name: str
    prefill_policy: str  # "Full_32L", "Uniform_k16", "Skip_AllTool_k16", "Skip_OldTool_k16"
    prefill_shallow_k: int
    prefill_align: AlignedPrefillConfig
    use_speculative_decode: bool
    decode_routing: RoutingConfig | None = None
    use_shortlist: bool = True
    max_chunk_len: int = 1024


@dataclass(frozen=True)
class UnifiedE2EResult:
    """Immutable end-to-end telemetry covering Prefill (TTFT), Decode, and Total wall-clock time."""

    config_name: str
    tokens: tuple[int, ...]
    prompt_len: int
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    prefill_tok_per_sec: float
    decode_tok_per_sec: float
    verify_calls: int
    mean_accepted_tokens: float
    total_neural_steps: int
    total_lookup_steps: int
    rollback_count: int


def baseline_e2e_config() -> UnifiedPipelineConfig:
    """Pure preset: standard 32-layer Prefill + standard 32-layer greedy Decode."""
    return UnifiedPipelineConfig(
        name="Baseline_E2E",
        prefill_policy="Full_32L",
        prefill_shallow_k=32,
        prefill_align=AlignedPrefillConfig(align_attn_kv=False, gdn_mode="none"),
        use_speculative_decode=False,
        decode_routing=None,
        use_shortlist=False,
    )


def decode_only_d1_config() -> UnifiedPipelineConfig:
    """Pure preset: exact 32L Prefill + Direction 1 PhaseGate_SL8K_NoCal Decode (lossless)."""
    return UnifiedPipelineConfig(
        name="DecodeOnly_D1",
        prefill_policy="Full_32L",
        prefill_shallow_k=32,
        prefill_align=AlignedPrefillConfig(align_attn_kv=False, gdn_mode="none"),
        use_speculative_decode=True,
        decode_routing=phase_gated_routing(confidence_threshold=0.70, max_neural_len=6),
        use_shortlist=True,
    )


def prefill_only_d2_config() -> UnifiedPipelineConfig:
    """Pure preset: Direction 2 U16_AttnKV Prefill + standard 32L Decode."""
    return UnifiedPipelineConfig(
        name="PrefillOnly_D2",
        prefill_policy="Uniform_k16",
        prefill_shallow_k=16,
        prefill_align=AlignedPrefillConfig(align_attn_kv=True, gdn_mode="none"),
        use_speculative_decode=False,
        decode_routing=None,
        use_shortlist=False,
    )


def unified_d2_d1_config() -> UnifiedPipelineConfig:
    """Pure preset: Direction 2 U16_AttnKV Prefill (1.61x) + Direction 1 PhaseGate_SL8K Decode (1.86x)."""
    return UnifiedPipelineConfig(
        name="Unified_D2_D1",
        prefill_policy="Uniform_k16",
        prefill_shallow_k=16,
        prefill_align=AlignedPrefillConfig(align_attn_kv=True, gdn_mode="none"),
        use_speculative_decode=True,
        decode_routing=phase_gated_routing(confidence_threshold=0.70, max_neural_len=6),
        use_shortlist=True,
    )


def unified_d23_d1_config() -> UnifiedPipelineConfig:
    """Pure preset: Direction 3+2 Role-Aware Skip_AllTool_k16 + AttnKV Prefill + Direction 1 Decode."""
    return UnifiedPipelineConfig(
        name="Unified_D23_D1",
        prefill_policy="Skip_AllTool_k16",
        prefill_shallow_k=16,
        prefill_align=AlignedPrefillConfig(align_attn_kv=True, gdn_mode="none"),
        use_speculative_decode=True,
        decode_routing=phase_gated_routing(confidence_threshold=0.70, max_neural_len=6),
        use_shortlist=True,
    )


class UnifiedSpeculativePipeline:
    """End-to-end Apple Silicon MLX inference pipeline unifying Prefill alignment and Speculative Decode."""

    def __init__(
        self,
        model,
        tokenizer,
        w8_weight: np.ndarray,
        w8_bias: np.ndarray,
        phase_tags: PhaseTagIds,
        static_shortlist: StaticShortlist,
        compiled_prefill_aligner: CompiledHybridAligner,
        draft_layer: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = len(model.layers)
        self.phase_tags = phase_tags
        self.compiled_prefill_aligner = compiled_prefill_aligner
        self.decode_engine = PhaseShortlistSpeculativeEngine(
            model=model,
            w8_weight=w8_weight,
            w8_bias=w8_bias,
            phase_tags=phase_tags,
            static_shortlist=static_shortlist,
            logz_calibrator=None,
            draft_layer=draft_layer,
        )

    def _run_prefill_stage(
        self,
        prompt_ids: Sequence[int],
        config: UnifiedPipelineConfig,
    ) -> tuple[list, int, float]:
        """Execute Prefill stage (Full 32L or Aligned Shallow/Role-Aware) and return `(cache, first_tok, prefill_s)`."""
        t0 = time.perf_counter()
        if config.prefill_policy == "Full_32L":
            cache = self.model.make_cache()
            # Chunked causal prefill for bounded memory on 16GB M5
            n = len(prompt_ids)
            last_logits = None
            for start in range(0, n, config.max_chunk_len):
                chunk = prompt_ids[start : start + config.max_chunk_len]
                out = self.model(mx.array([list(chunk)], dtype=mx.int32), cache=cache)
                last_logits = out[0, -1, :]
                mx.eval(last_logits)
            assert last_logits is not None
            first_tok = int(mx.argmax(last_logits, axis=-1).item())
            return cache, first_tok, time.perf_counter() - t0

        # Shallow / Role-Aware Aligned Prefill on prompt_ids[:-1], then full 32L on final prompt token for exact first_tok
        prefix_ids = prompt_ids[:-1]
        prefix_texts = [self.tokenizer.decode([int(tid)]) for tid in prefix_ids]
        prefix_roles = parse_role_tags(prefix_texts, token_ids=prefix_ids)
        sched = build_depth_schedule(
            prefix_ids,
            prefix_roles,
            policy_name=config.prefill_policy,
            shallow_k=config.prefill_shallow_k,
            full_depth=self.num_layers,
            max_chunk_len=config.max_chunk_len,
        )
        cache = run_aligned_shallow_prefill(
            self.model,
            sched.chunks,
            compiled=self.compiled_prefill_aligner,
            config=config.prefill_align,
        )
        last_logits = self.model(mx.array([[int(prompt_ids[-1])]], dtype=mx.int32), cache=cache)[0, -1, :]
        next_tok = mx.argmax(last_logits, axis=-1)
        mx.eval(next_tok)
        return cache, int(next_tok.item()), time.perf_counter() - t0

    def _run_standard_decode(
        self,
        cache: list,
        first_tok: int,
        max_new_tokens: int,
        eos_set: frozenset[int],
    ) -> tuple[tuple[int, ...], float]:
        cur_tok = first_tok
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
        return tuple(generated), time.perf_counter() - t0

    def _run_speculative_decode_from_cache(
        self,
        cache: list,
        prompt_ids: Sequence[int],
        first_tok: int,
        max_new_tokens: int,
        eos_set: frozenset[int],
        routing: RoutingConfig,
        use_shortlist: bool,
    ) -> tuple[tuple[int, ...], float, int, int, int, int]:
        """Run Direction 1 PhaseGate + Shortlist speculative decode directly on a pre-populated cache."""
        eng = self.decode_engine
        prompt_sub_head: QuantizedSubHead | None = None
        if use_shortlist and eng.static_shortlist is not None and eng.static_sub_head is not None:
            prompt_sub_head, _ = build_prompt_sub_head(
                eng.text_model.embed_tokens,
                eng.static_shortlist,
                eng.static_sub_head,
                prompt_ids,
                static_token_set=eng.static_token_set,
            )

        history_buf = [int(t) for t in prompt_ids]
        generated: list[int] = []
        phase = infer_phase_from_tokens(prompt_ids, self.phase_tags)
        cur_tok = first_tok

        verify_calls = 0
        total_neural = 0
        total_lookup = 0
        rollbacks = 0

        t0 = time.perf_counter()
        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            if cur_tok in eos_set or remaining == 1:
                generated.append(cur_tok)
                break

            generated.append(cur_tok)
            history_buf.append(cur_tok)
            phase = step_phase(phase, cur_tok, self.phase_tags)
            remaining -= 1

            policy = policy_for_phase(routing, phase)
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
                    filtered = _truncate_at_first_eos(copied, eos_set)
                    draft_tokens.extend(filtered)
                    lookup_steps = len(filtered)

            draft_gdn_snaps: list[list[tuple[mx.array | None, mx.array | None]]] = []
            draft_hiddens: list[mx.array] = []

            if policy.use_neural and lookup_steps == 0 and policy.max_neural_len > 0:
                feed_tok = cur_tok
                draft_phase = phase
                limit = min(policy.max_neural_len, remaining)
                while neural_steps < limit:
                    h8 = eng._run_draft_layers_one_token(feed_tok, cache)
                    draft_hiddens.append(h8)
                    draft_gdn_snaps.append(
                        [(cache[i][0], cache[i][1]) for i in range(eng.draft_layer) if self.model.layers[i].is_linear]
                    )
                    pred = eng._draft_predict(
                        h8,
                        prompt_sub_head=prompt_sub_head,
                        use_shortlist=use_shortlist,
                        use_logz_calibration=False,
                    )
                    draft_tokens.append(pred.token_id)
                    neural_steps += 1
                    feed_tok = pred.token_id
                    if pred.token_id in eos_set:
                        break
                    if policy.confidence_threshold > 0.0 and pred.calibrated_prob < policy.confidence_threshold:
                        break
                    next_dp = step_phase(draft_phase, pred.token_id, self.phase_tags)
                    if not policy_for_phase(routing, next_dp).use_neural:
                        break
                    draft_phase = next_dp

            verify_calls += 1
            total_neural += neural_steps
            total_lookup += lookup_steps

            if neural_steps > 0:
                h8_chunk = mx.concatenate(draft_hiddens, axis=1)
                k_total = neural_steps
                verify_logits, gdn_stashes = eng._run_verify_layers(
                    h8_chunk, cache, start_layer=eng.draft_layer, stash_gdn=(k_total > 1)
                )
            else:
                verify_inputs = [cur_tok] + draft_tokens[:-1]
                k_total = len(verify_inputs)
                x_emb = eng.text_model.embed_tokens(mx.array([verify_inputs], dtype=mx.int32))
                verify_logits, gdn_stashes = eng._run_verify_layers(
                    x_emb, cache, start_layer=0, stash_gdn=(k_total > 1)
                )

            target_toks_mx = mx.argmax(verify_logits, axis=-1)
            mx.eval(target_toks_mx)
            target_toks = [int(x) for x in target_toks_mx.tolist()]

            matched = 0
            for d_tok, t_tok in zip(draft_tokens, target_toks):
                if d_tok == t_tok:
                    matched += 1
                else:
                    break

            n_keep = min(matched + 1, k_total)
            if n_keep < k_total:
                rollbacks += 1
                drop_kv = k_total - n_keep
                for i in range(self.num_layers):
                    if not self.model.layers[i].is_linear:
                        cache[i].trim(drop_kv)
                if neural_steps > 0:
                    lin_idx = 0
                    for i in range(eng.draft_layer):
                        if self.model.layers[i].is_linear:
                            cache[i][0], cache[i][1] = draft_gdn_snaps[n_keep - 1][lin_idx]
                            lin_idx += 1
                for st in gdn_stashes:
                    _rollback_gdn_from_stash(self.model.layers[st.layer_idx], cache[st.layer_idx], st, n_keep)
                mx.eval([c[1] for layer, c in zip(self.model.layers, cache) if layer.is_linear and c[1] is not None])

            if matched > 0:
                all_matched = matched == len(draft_tokens)
                commit_count = matched - 1 if all_matched else matched
                for idx in range(matched):
                    tok = draft_tokens[idx]
                    if tok in eos_set:
                        generated.append(tok)
                        return tuple(generated), time.perf_counter() - t0, verify_calls, total_neural, total_lookup, rollbacks
                    if idx < commit_count:
                        generated.append(tok)
                        history_buf.append(tok)
                        phase = step_phase(phase, tok, self.phase_tags)
                cur_tok = draft_tokens[-1] if all_matched else target_toks[matched]
            else:
                cur_tok = target_toks[0]

        return tuple(generated), time.perf_counter() - t0, verify_calls, total_neural, total_lookup, rollbacks

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_ids: Sequence[int] | set[int] | frozenset[int],
        config: UnifiedPipelineConfig,
    ) -> UnifiedE2EResult:
        """Run full Prefill -> Decode end-to-end generation and return immutable wall-clock telemetry."""
        eos_set = frozenset(int(t) for t in eos_token_ids)
        cache, first_tok, prefill_s = self._run_prefill_stage(prompt_ids, config)

        if config.use_speculative_decode and config.decode_routing is not None:
            tokens, decode_s, vcalls, n_steps, l_steps, rolls = self._run_speculative_decode_from_cache(
                cache=cache,
                prompt_ids=prompt_ids,
                first_tok=first_tok,
                max_new_tokens=max_new_tokens,
                eos_set=eos_set,
                routing=config.decode_routing,
                use_shortlist=config.use_shortlist,
            )
        else:
            tokens, decode_s = self._run_standard_decode(cache, first_tok, max_new_tokens, eos_set)
            vcalls = len(tokens)
            n_steps = 0
            l_steps = 0
            rolls = 0

        del cache
        mx.clear_cache()

        total_s = prefill_s + decode_s
        return UnifiedE2EResult(
            config_name=config.name,
            tokens=tokens,
            prompt_len=len(prompt_ids),
            prefill_seconds=prefill_s,
            decode_seconds=decode_s,
            total_seconds=total_s,
            prefill_tok_per_sec=len(prompt_ids) / prefill_s if prefill_s > 0 else 0.0,
            decode_tok_per_sec=len(tokens) / decode_s if decode_s > 0 else 0.0,
            verify_calls=vcalls,
            mean_accepted_tokens=len(tokens) / vcalls if vcalls > 0 else 1.0,
            total_neural_steps=n_steps,
            total_lookup_steps=l_steps,
            rollback_count=rolls,
        )
