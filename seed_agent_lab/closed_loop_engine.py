"""True closed-loop speculative engine on Qwen3.5-4B: Layer-8 Ridge + Prompt Lookup + GatedDeltaNet rollback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

from seed_agent_lab.hybrid_speculative import find_prompt_lookup_draft


@dataclass(frozen=True)
class ClosedLoopResult:
    """Generated token IDs and detailed wall-clock / acceptance statistics."""

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


@dataclass
class _VerifyGdnStash:
    """Transient per-layer tensors kept during one Verify pass for O(1)-projection rollback."""

    layer_idx: int
    pre_state: mx.array | None
    conv_input: mx.array
    q: mx.array
    k: mx.array
    v: mx.array
    a: mx.array
    b: mx.array
    n_keep_conv: int


def _gdn_step_forward(layer_module, x_hidden: mx.array, cache, stash_out: bool, layer_idx: int) -> tuple[mx.array, _VerifyGdnStash | None]:
    """Run DecoderLayer for a linear-attention layer, optionally stashing Verify tensors for rollback."""
    gdn = layer_module.linear_attn
    x_normed = layer_module.input_layernorm(x_hidden)
    B, S, _ = x_normed.shape

    qkv = gdn.in_proj_qkv(x_normed)
    z = gdn.in_proj_z(x_normed).reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)
    b = gdn.in_proj_b(x_normed)
    a = gdn.in_proj_a(x_normed)

    if cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=x_normed.dtype)

    n_keep_conv = gdn.conv_kernel_size - 1
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    cache[0] = mx.contiguous(conv_input[:, -n_keep_conv:, :])
    conv_out = nn.silu(gdn.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
        )
    ]

    pre_state = cache[1]
    q, k = normalize_qk(q, k, inv_scale=gdn.head_k_dim**-0.5, eps=1e-6)
    out, new_state = gated_delta_update(
        q, k, v, a, b, gdn.A_log, gdn.dt_bias, pre_state, mask=None, use_kernel=True
    )
    cache[1] = new_state
    cache.advance(S)

    out = gdn.norm(out, z)
    r = gdn.out_proj(out.reshape(B, S, -1))
    h = x_hidden + r
    out_hidden = h + layer_module.mlp(layer_module.post_attention_layernorm(h))

    stash = None
    if stash_out:
        stash = _VerifyGdnStash(
            layer_idx=layer_idx,
            pre_state=pre_state,
            conv_input=conv_input,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            n_keep_conv=n_keep_conv,
        )
    return out_hidden, stash


def _rollback_gdn_from_stash(layer_module, cache, stash: _VerifyGdnStash, n_keep: int) -> None:
    """Recompute exact GatedDeltaNet state after n_keep tokens using stashed Verify projections."""
    gdn = layer_module.linear_attn
    cache[0] = mx.contiguous(stash.conv_input[:, n_keep : n_keep + stash.n_keep_conv, :])
    _, rolled_state = gated_delta_update(
        stash.q[:, :n_keep],
        stash.k[:, :n_keep],
        stash.v[:, :n_keep],
        stash.a[:, :n_keep],
        stash.b[:, :n_keep],
        gdn.A_log,
        gdn.dt_bias,
        stash.pre_state,
        mask=None,
        use_kernel=True,
    )
    cache[1] = rolled_state


class ClosedLoopSpeculativeEngine:
    """Closed-loop greedy speculative decoder on Qwen3.5-4B with exact state/cache rollback."""

    def __init__(
        self,
        model,
        w8_weight: np.ndarray,
        w8_bias: np.ndarray,
        draft_layer: int = 8,
    ) -> None:
        self.model = model
        self.text_model = model.language_model.model
        self.lm = model.language_model
        self.draft_layer = draft_layer
        self.num_layers = len(model.layers)

        # Match Qwen3.5 native bfloat16 activation/LM-head dtype on Metal GPU.
        self.w8_weight = mx.array(w8_weight, dtype=mx.bfloat16)  # (D, D)
        self.w8_bias = mx.array(w8_bias, dtype=mx.bfloat16)  # (1, D)
        mx.eval(self.w8_weight, self.w8_bias)

        self.draft_fa_idx = next((i for i in range(draft_layer) if not model.layers[i].is_linear), None)
        self.verify_fa_idx = next((i for i in range(draft_layer, self.num_layers) if not model.layers[i].is_linear), None)

    def _lm_head(self, normed_hidden: mx.array) -> mx.array:
        if self.lm.args.tie_word_embeddings:
            return self.text_model.embed_tokens.as_linear(normed_hidden)
        return self.lm.lm_head(normed_hidden)

    def _draft_predict(self, hidden_at_k: mx.array) -> tuple[int, float]:
        normed = self.text_model.norm(hidden_at_k)
        rotated = (normed @ self.w8_weight) + self.w8_bias
        logits = self._lm_head(rotated)[0, -1, :]
        # Compute argmax token and scalar max softmax probability cleanly.
        logits_f32 = logits.astype(mx.float32)
        tok = mx.argmax(logits, axis=-1)
        max_prob = mx.exp(mx.max(logits_f32, axis=-1) - mx.logsumexp(logits_f32, axis=-1))
        mx.eval(tok, max_prob)
        return int(tok.item()), float(max_prob.item())

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
        logits = self._lm_head(self.text_model.norm(hidden))[0]  # (S, V)
        return logits, stashes

    def generate_baseline(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_ids: set[int],
    ) -> tuple[tuple[int, ...], float]:
        """Standard full 32-layer greedy baseline using identical model & cache."""
        import time

        cache = self.model.make_cache()
        logits = self.model(mx.array([list(prompt_ids)], dtype=mx.int32), cache=cache)[0, -1, :]
        next_tok = mx.argmax(logits, axis=-1)
        mx.eval(next_tok)
        cur_tok = int(next_tok.item())

        generated: list[int] = []
        t0 = time.perf_counter()
        while len(generated) < max_new_tokens:
            generated.append(cur_tok)
            if cur_tok in eos_token_ids or len(generated) >= max_new_tokens:
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
        eos_token_ids: set[int],
        use_neural: bool = True,
        use_lookup: bool = True,
        lookup_first: bool = True,
        max_neural_len: int = 6,
        max_lookup_len: int = 8,
        ngram_size: int = 3,
        confidence_threshold: float = 0.70,
    ) -> ClosedLoopResult:
        """True closed-loop speculative generation matching greedy baseline token-for-token."""
        import time

        cache = self.model.make_cache()
        logits = self.model(mx.array([list(prompt_ids)], dtype=mx.int32), cache=cache)[0, -1, :]
        next_tok = mx.argmax(logits, axis=-1)
        mx.eval(next_tok)
        cur_tok = int(next_tok.item())

        history_buf = list(prompt_ids)
        generated: list[int] = []

        verify_calls = 0
        total_neural_steps = 0
        total_lookup_steps = 0
        rollback_count = 0
        draft_seconds = 0.0
        verify_seconds = 0.0
        rollback_seconds = 0.0

        t_start = time.perf_counter()
        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            # cur_tok is ALREADY verified for the current position!
            # Check if we already hit EOS or max_new_tokens with just cur_tok.
            if cur_tok in eos_token_ids or remaining == 1:
                generated.append(cur_tok)
                break

            # Otherwise, cur_tok is emitted after verification or directly; we need to predict tokens AFTER cur_tok.
            history_buf.append(cur_tok)
            generated.append(cur_tok)
            remaining -= 1

            td0 = time.perf_counter()
            draft_tokens: list[int] = []
            lookup_steps = 0
            neural_steps = 0

            if use_lookup and lookup_first:
                copied = find_prompt_lookup_draft(
                    history_buf,
                    ngram_size=ngram_size,
                    max_draft_len=min(max_lookup_len, remaining),
                )
                if copied:
                    draft_tokens.extend(copied)
                    lookup_steps = len(copied)

            draft_gdn_snapshots: list[list[tuple[mx.array | None, mx.array | None]]] = []
            draft_hiddens: list[mx.array] = []

            if use_neural and lookup_steps == 0:
                feed_tok = cur_tok
                limit = min(max_neural_len, remaining)
                while neural_steps < limit:
                    h8 = self._run_draft_layers_one_token(feed_tok, cache)
                    draft_hiddens.append(h8)
                    # Snapshot immutable MLX array refs for the 6 draft GatedDeltaNet layers (zero copy).
                    snap = [
                        (cache[i][0], cache[i][1])
                        for i in range(self.draft_layer)
                        if self.model.layers[i].is_linear
                    ]
                    draft_gdn_snapshots.append(snap)

                    d_tok, d_prob = self._draft_predict(h8)
                    draft_tokens.append(d_tok)
                    neural_steps += 1
                    feed_tok = d_tok
                    if confidence_threshold > 0.0 and d_prob < confidence_threshold:
                        break
            draft_seconds += time.perf_counter() - td0

            tv0 = time.perf_counter()
            verify_calls += 1
            total_neural_steps += neural_steps
            total_lookup_steps += lookup_steps

            if neural_steps > 0:
                # Verify drafted tokens in parallel using layers 8..31 only!
                h8_chunk = mx.concatenate(draft_hiddens, axis=1)  # (1, neural_steps, D)
                k_total = neural_steps
                verify_logits, gdn_stashes = self._run_verify_layers(
                    h8_chunk, cache, start_layer=self.draft_layer, stash_gdn=(k_total > 1)
                )
            else:
                # Lookup burst (or single step if neither neural nor lookup ran): run all 32 layers!
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

            # Compare draft_tokens against target_toks.
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

                if neural_steps > 0:
                    # Trim KVCaches for ALL 32 layers, restore draft GDN refs for 0..k-1, and slice-rollback 8..31.
                    for i in range(self.num_layers):
                        if not self.model.layers[i].is_linear:
                            cache[i].trim(drop_kv)
                    lin_idx = 0
                    for i in range(self.draft_layer):
                        if self.model.layers[i].is_linear:
                            cache[i][0], cache[i][1] = draft_gdn_snapshots[n_keep - 1][lin_idx]
                            lin_idx += 1
                    for st in gdn_stashes:
                        _rollback_gdn_from_stash(self.model.layers[st.layer_idx], cache[st.layer_idx], st, n_keep)
                else:
                    # Lookup burst ran layers 0..31 -> trim KVCache and slice-rollback all 24 GDN stashes.
                    for i in range(self.num_layers):
                        if not self.model.layers[i].is_linear:
                            cache[i].trim(drop_kv)
                    for st in gdn_stashes:
                        _rollback_gdn_from_stash(self.model.layers[st.layer_idx], cache[st.layer_idx], st, n_keep)

                mx.eval([c[1] for l, c in zip(self.model.layers, cache) if l.is_linear and c[1] is not None])
                rollback_seconds += time.perf_counter() - tr0

            # Commit matched tokens and set next cur_tok!
            if matched > 0:
                accepted_slice = draft_tokens[:matched]
                for tok in accepted_slice:
                    if tok in eos_token_ids:
                        generated.append(tok)
                        elapsed = time.perf_counter() - t_start
                        del cache
                        mx.clear_cache()
                        return ClosedLoopResult(
                            tokens=tuple(generated),
                            elapsed_seconds=elapsed,
                            tokens_per_sec=len(generated) / elapsed if elapsed > 0 else 0.0,
                            verify_calls=verify_calls,
                            mean_accepted_tokens=len(generated) / verify_calls,
                            total_neural_steps=total_neural_steps,
                            total_lookup_steps=total_lookup_steps,
                            rollback_count=rollback_count,
                            draft_seconds=draft_seconds,
                            verify_seconds=verify_seconds,
                            rollback_seconds=rollback_seconds,
                        )
                    generated.append(tok)
                    history_buf.append(tok)

            if not draft_tokens or matched < len(draft_tokens):
                cur_tok = target_toks[matched]
            else:
                # All drafted tokens matched! Notice cur_tok already verified up to draft_tokens[-1],
                # and draft_tokens[-1] was NOT fed into caches yet (caches hold cur_tok..draft_tokens[-2]).
                # Wait: because we ALREADY appended all draft_tokens[:matched] to generated,
                # the LAST accepted token draft_tokens[-1] hasn't been fed into caches to produce the NEXT token!
                # Let's check!
                # Wait: in our next loop iteration, we append `cur_tok` at the top of `while`!
                # So if we set `cur_tok = draft_tokens[-1]` AND DON'T append `draft_tokens[-1]` twice:
                generated.pop()
                history_buf.pop()
                cur_tok = draft_tokens[-1]

        elapsed = time.perf_counter() - t_start
        del cache
        mx.clear_cache()
        return ClosedLoopResult(
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
        )
