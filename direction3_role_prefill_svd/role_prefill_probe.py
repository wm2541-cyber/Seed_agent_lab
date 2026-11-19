"""Controlled Role-Aware Shallow Prefill, GDN State-Error SVD, and Direction-2 Causal Oracle probes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache
import numpy as np

from direction3_role_prefill_svd.role_segments import (
    DepthSchedule,
    TokenRole,
    build_depth_schedule,
    parse_role_tags,
)
from direction3_role_prefill_svd.svd_analysis import (
    StateErrorDecomposition,
    decompose_state_error,
    low_rank_replace_content_error,
)
from seed_agent_lab.analysis import compute_layer_exit_metrics
from seed_agent_lab.run_seed_coding_eval import (
    _clone_cache,
    _find_last_assistant_span,
    _run_chunks_with_shallow_prefill,
    _score_continuation,
)
from seed_agent_lab.run_swe_rotation_probe import RunningMetrics, _prepare_trajectory


@dataclass(frozen=True)
class PolicyEvalSpec:
    """Immutable specification of one role-aware shallow prefill policy."""

    name: str
    shallow_k: int


@dataclass(frozen=True)
class PolicyAggregateMetrics:
    """Immutable aggregated evaluation metrics for one prefill or oracle policy."""

    policy_name: str
    mean_skipped_layer_frac: float
    mean_skipped_tokens: float
    mean_skipped_rel_pos: float
    all_asst_n: int
    all_asst_kl: float
    all_asst_top1: float
    all_asst_top5: float
    by_category: tuple[tuple[str, tuple[int, float, float]], ...]  # (cat, (n, kl, top1))


@dataclass(frozen=True)
class DeepLayerStateErrorSummary:
    """Immutable layer-level GDN state error & E_ax SVD summary aggregated across heads and traces."""

    policy_name: str
    layer_idx: int
    n_head_samples: int
    mean_sink_err_share: float
    mean_content_err_share: float
    mean_rel_sink_err: float
    mean_rel_content_err: float
    mean_eax_eff_rank: float
    mean_eax_r50: float
    mean_eax_r80: float
    mean_eax_r90: float
    mean_eax_r95: float
    mean_eax_top1_energy: float
    mean_eax_top8_energy: float
    mean_eax_top16_energy: float


DEFAULT_PREFILL_POLICIES: tuple[PolicyEvalSpec, ...] = (
    PolicyEvalSpec(name="Uniform_k16", shallow_k=16),
    PolicyEvalSpec(name="Uniform_k24", shallow_k=24),
    PolicyEvalSpec(name="Adaptive_old16_recent24", shallow_k=16),
    PolicyEvalSpec(name="Skip_AllTool_k16", shallow_k=16),
    PolicyEvalSpec(name="Skip_OldTool_k16", shallow_k=16),
    PolicyEvalSpec(name="Skip_AllTool_k20", shallow_k=20),
    PolicyEvalSpec(name="Matched_SkipTool_k16", shallow_k=16),
    PolicyEvalSpec(name="Matched_SkipUserAsst_k16", shallow_k=16),
    PolicyEvalSpec(name="Matched_SkipUserAsstRecent_k16", shallow_k=16),
)


def _update_bucket_metrics(
    buckets: dict[str, RunningMetrics],
    tag_buckets: Mapping[str, Sequence[int]],
    cand_logits: np.ndarray,
    exact_logits: np.ndarray,
) -> None:
    """Update exact-vs-candidate KL/top1/top5 immutable RunningMetrics across assistant token buckets."""
    for bucket_name, positions in tag_buckets.items():
        if positions:
            idx = list(positions)
            batch = compute_layer_exit_metrics(0, cand_logits[idx], exact_logits[idx])
            buckets[bucket_name] = buckets[bucket_name].add(batch)


def _build_surgical_oracle_cache(
    model,
    shallow_cache: Sequence,
    exact_cache: Sequence,
    mean_v: Mapping[int, np.ndarray],
    mean_a: Mapping[int, np.ndarray],
    start_layer: int,
    restore_attn_kv: bool,
    restore_full_gdn: bool,
    use_mean_sink_b: bool,
    oracle_ax_rank: int | None,
) -> list:
    """Create a cloned cache selectively restoring deep Attention KV and/or deep GDN (b + Rank-r ax)."""
    merged = _clone_cache(shallow_cache)
    for idx in range(start_layer, len(model.layers)):
        layer = model.layers[idx]
        if not layer.is_linear:
            if restore_attn_kv:
                src_kv = exact_cache[idx]
                dst_kv = KVCache()
                if src_kv.keys is not None and src_kv.values is not None:
                    dst_kv.keys = mx.array(src_kv.keys)
                    dst_kv.values = mx.array(src_kv.values)
                    dst_kv.offset = int(src_kv.offset)
                merged[idx] = dst_kv
        else:
            if restore_full_gdn:
                src_arr = exact_cache[idx]
                dst_arr = ArraysCache(len(src_arr.cache))
                dst_arr.cache = [mx.array(a) if a is not None else None for a in src_arr.cache]
                merged[idx] = dst_arr
            elif use_mean_sink_b or oracle_ax_rank is not None:
                sh_state = merged[idx][1]
                ex_state = exact_cache[idx][1]
                if sh_state is not None and ex_state is not None:
                    orig_dtype = sh_state.dtype
                    sh_np = np.array(sh_state[0].astype(mx.float32), copy=True)
                    ex_np = np.array(ex_state[0].astype(mx.float32), copy=True)
                    rank = int(oracle_ax_rank) if oracle_ax_rank is not None else 0
                    new_heads = np.stack(
                        [
                            low_rank_replace_content_error(
                                true_state_dv_dk=ex_np[h],
                                shallow_state_dv_dk=sh_np[h],
                                mean_v=mean_v[idx][h],
                                rank=rank,
                                mean_sink_coeff=mean_a[idx][h] if use_mean_sink_b else None,
                            )
                            for h in range(sh_np.shape[0])
                        ],
                        axis=0,
                    )
                    merged[idx][1] = mx.array(new_heads[None, ...], dtype=orig_dtype)
    mx.eval(
        [
            x
            for c in merged
            for x in (c.cache if isinstance(c, ArraysCache) else (c.keys, c.values))
            if x is not None
        ]
    )
    return merged


def _collect_deep_gdn_state_errors(
    model,
    exact_cache: Sequence,
    shallow_cache: Sequence,
    mean_v: Mapping[int, np.ndarray],
    start_layer: int,
) -> dict[int, list[StateErrorDecomposition]]:
    """Compare deep GDN recurrent states S_true vs S_shallow across all heads of layers >= start_layer."""
    per_layer: dict[int, list[StateErrorDecomposition]] = {}
    for idx in range(start_layer, len(model.layers)):
        layer = model.layers[idx]
        if not layer.is_linear:
            continue
        ex_st = exact_cache[idx][1]
        sh_st = shallow_cache[idx][1]
        if ex_st is None or sh_st is None:
            continue
        mx.eval(ex_st, sh_st)
        ex_np = np.array(ex_st[0].astype(mx.float32), copy=True)
        sh_np = np.array(sh_st[0].astype(mx.float32), copy=True)
        per_layer[idx] = [
            decompose_state_error(ex_np[h], sh_np[h], mean_v[idx][h])
            for h in range(ex_np.shape[0])
        ]
    return per_layer


def _finalize_policy_metrics(
    policy_name: str,
    buckets: Mapping[str, RunningMetrics],
    schedules: Sequence[DepthSchedule],
) -> PolicyAggregateMetrics:
    """Purely summarize RunningMetrics + DepthSchedules into an immutable PolicyAggregateMetrics."""
    all_m = buckets["all_asst"].finalize(0)
    cats: list[tuple[str, tuple[int, float, float]]] = []
    for cname in ("reasoning", "new_code_arg", "tool_boilerplate", "repeated_echo"):
        cm = buckets.get(cname)
        if cm is not None and cm.count > 0:
            fin = cm.finalize(0)
            cats.append((cname, (fin.count, fin.mean_kl_nats, fin.top1_match_rate)))
    return PolicyAggregateMetrics(
        policy_name=policy_name,
        mean_skipped_layer_frac=float(np.mean([s.skipped_layer_fraction for s in schedules])) if schedules else 0.0,
        mean_skipped_tokens=float(np.mean([s.skipped_token_count for s in schedules])) if schedules else 0.0,
        mean_skipped_rel_pos=float(np.mean([s.mean_skipped_rel_pos for s in schedules])) if schedules else 0.0,
        all_asst_n=all_m.count,
        all_asst_kl=all_m.mean_kl_nats,
        all_asst_top1=all_m.top1_match_rate,
        all_asst_top5=all_m.top5_match_rate,
        by_category=tuple(cats),
    )


def _finalize_state_error_summaries(
    policy_name: str,
    accumulated: Mapping[int, Sequence[StateErrorDecomposition]],
) -> tuple[DeepLayerStateErrorSummary, ...]:
    """Purely aggregate per-head StateErrorDecompositions by layer index."""
    out: list[DeepLayerStateErrorSummary] = []
    for layer_idx in sorted(accumulated.keys()):
        items = accumulated[layer_idx]
        if not items:
            continue
        out.append(
            DeepLayerStateErrorSummary(
                policy_name=policy_name,
                layer_idx=layer_idx,
                n_head_samples=len(items),
                mean_sink_err_share=float(np.mean([d.sink_err_share for d in items])),
                mean_content_err_share=float(np.mean([d.content_err_share for d in items])),
                mean_rel_sink_err=float(np.mean([d.rel_sink_err for d in items])),
                mean_rel_content_err=float(np.mean([d.rel_content_err for d in items])),
                mean_eax_eff_rank=float(np.mean([d.content_err_spectrum.effective_rank for d in items])),
                mean_eax_r50=float(np.mean([d.content_err_spectrum.r50 for d in items])),
                mean_eax_r80=float(np.mean([d.content_err_spectrum.r80 for d in items])),
                mean_eax_r90=float(np.mean([d.content_err_spectrum.r90 for d in items])),
                mean_eax_r95=float(np.mean([d.content_err_spectrum.r95 for d in items])),
                mean_eax_top1_energy=float(np.mean([d.content_err_spectrum.top1_energy for d in items])),
                mean_eax_top8_energy=float(np.mean([d.content_err_spectrum.top8_energy for d in items])),
                mean_eax_top16_energy=float(np.mean([d.content_err_spectrum.top16_energy for d in items])),
            )
        )
    return tuple(out)


def run_role_prefill_and_state_svd_probe(
    model,
    tokenizer,
    held_out_rows: Sequence[dict],
    mean_v: Mapping[int, np.ndarray],
    mean_a: Mapping[int, np.ndarray],
    max_tokens: int = 6144,
    max_chunk_len: int = 1024,
) -> tuple[
    tuple[PolicyAggregateMetrics, ...],
    tuple[PolicyAggregateMetrics, ...],
    dict[str, tuple[DeepLayerStateErrorSummary, ...]],
    list[dict],
]:
    """Run Probe 1 (Role-Aware Prefill), Probe 2A (GDN State-Error SVD), and Probe 2B (Direction-2 Oracle Ceiling)."""
    num_layers = len(model.layers)

    prefill_metrics: dict[str, dict[str, RunningMetrics]] = defaultdict(lambda: defaultdict(RunningMetrics))
    prefill_schedules: dict[str, list[DepthSchedule]] = defaultdict(list)

    oracle_metrics: dict[str, dict[str, RunningMetrics]] = defaultdict(lambda: defaultdict(RunningMetrics))
    oracle_schedules: dict[str, list[DepthSchedule]] = defaultdict(list)

    state_err_targets = ("Uniform_k16", "Skip_AllTool_k16", "Matched_SkipTool_k16", "Matched_SkipUserAsst_k16")
    state_err_acc: dict[str, dict[int, list[StateErrorDecomposition]]] = {
        name: defaultdict(list) for name in state_err_targets
    }

    trace_meta: list[dict] = []

    for offset, row in enumerate(held_out_rows):
        row_idx = 12 + offset
        item = _prepare_trajectory(tokenizer, row, max_tokens=max_tokens)
        if item is None:
            continue
        token_ids, tags, asst_mask = item
        span = _find_last_assistant_span(asst_mask, min_prompt_len=256)
        if span is None:
            continue
        split_pt, span_end = span
        prompt_prefix = token_ids[: split_pt - 1]
        cont_inputs = token_ids[split_pt - 1 : span_end - 1]
        cont_tags = tags[split_pt:span_end]
        cont_asst = asst_mask[split_pt:span_end]
        if not cont_inputs:
            continue

        prefix_texts = [tokenizer.decode([tid]) for tid in prompt_prefix]
        prefix_roles = parse_role_tags(prefix_texts, token_ids=prompt_prefix)
        role_counts = {r.value: sum(1 for x in prefix_roles if x is r) for r in TokenRole}
        trace_meta.append(
            {
                "row_idx": row_idx,
                "prefix_len": len(prompt_prefix),
                "cont_len": len(cont_inputs),
                "role_counts": role_counts,
                "tool_fraction": role_counts["tool"] / len(prompt_prefix),
            }
        )

        # Exact 32L teacher prefill & continuation logits
        full_sched = build_depth_schedule(
            prompt_prefix, prefix_roles, "Full_32L", full_depth=num_layers, max_chunk_len=max_chunk_len
        )
        exact_prompt_cache = _run_chunks_with_shallow_prefill(model, full_sched.chunks)
        exact_cont_cache = _clone_cache(exact_prompt_cache)
        exact_logits = _score_continuation(model, exact_cont_cache, cont_inputs)
        del exact_cont_cache

        tag_buckets: dict[str, list[int]] = defaultdict(list)
        for pos, (tag, is_asst) in enumerate(zip(cont_tags, cont_asst)):
            if is_asst:
                tag_buckets["all_asst"].append(pos)
                if tag in ("tool_boilerplate", "repeated_echo", "new_code_arg", "reasoning"):
                    tag_buckets[tag].append(pos)

        uniform_k16_cache = None
        uniform_k16_sched = None

        for spec in DEFAULT_PREFILL_POLICIES:
            sched = build_depth_schedule(
                prompt_prefix,
                prefix_roles,
                policy_name=spec.name,
                shallow_k=spec.shallow_k,
                full_depth=num_layers,
                max_chunk_len=max_chunk_len,
            )
            prefill_schedules[spec.name].append(sched)
            cand_cache = _run_chunks_with_shallow_prefill(model, sched.chunks)

            if spec.name in state_err_acc:
                errs = _collect_deep_gdn_state_errors(
                    model, exact_prompt_cache, cand_cache, mean_v=mean_v, start_layer=16
                )
                for l_idx, head_errs in errs.items():
                    state_err_acc[spec.name][l_idx].extend(head_errs)

            if spec.name == "Uniform_k16":
                uniform_k16_cache = _clone_cache(cand_cache)
                uniform_k16_sched = sched

            cand_logits = _score_continuation(model, cand_cache, cont_inputs)
            _update_bucket_metrics(prefill_metrics[spec.name], tag_buckets, cand_logits, exact_logits)
            del cand_cache, cand_logits
            mx.clear_cache()

        # --- Probe 2B: Direction-2 Causal Oracle Attribution on Uniform_k16 ---
        if uniform_k16_cache is not None and uniform_k16_sched is not None:
            oracle_specs = (
                ("Oracle_GDN_Only", False, True, False, None),
                ("Oracle_AttnKV_Only", True, False, False, None),
                ("Uniform_k16 + MeanSink_b", False, False, True, 0),
                ("Uniform_k16 + MeanSink_b + ax_R8", False, False, True, 8),
                ("Uniform_k16 + MeanSink_b + ax_R16", False, False, True, 16),
                ("Oracle_AttnKV + MeanSink_b", True, False, True, 0),
                ("Oracle_AttnKV + MeanSink_b + ax_R8", True, False, True, 8),
                ("Oracle_AttnKV + MeanSink_b + ax_R16", True, False, True, 16),
            )
            for oname, r_attn, r_gdn, use_mb, ax_r in oracle_specs:
                oracle_schedules[oname].append(uniform_k16_sched)
                ocache = _build_surgical_oracle_cache(
                    model=model,
                    shallow_cache=uniform_k16_cache,
                    exact_cache=exact_prompt_cache,
                    mean_v=mean_v,
                    mean_a=mean_a,
                    start_layer=16,
                    restore_attn_kv=r_attn,
                    restore_full_gdn=r_gdn,
                    use_mean_sink_b=use_mb,
                    oracle_ax_rank=ax_r,
                )
                ologits = _score_continuation(model, ocache, cont_inputs)
                _update_bucket_metrics(oracle_metrics[oname], tag_buckets, ologits, exact_logits)
                del ocache, ologits
                mx.clear_cache()
            del uniform_k16_cache

        del exact_prompt_cache, exact_logits
        mx.clear_cache()

    prefill_summary = tuple(
        _finalize_policy_metrics(spec.name, prefill_metrics[spec.name], prefill_schedules[spec.name])
        for spec in DEFAULT_PREFILL_POLICIES
        if prefill_metrics[spec.name]["all_asst"].count > 0
    )
    oracle_order = (
        "Oracle_GDN_Only",
        "Oracle_AttnKV_Only",
        "Uniform_k16 + MeanSink_b",
        "Uniform_k16 + MeanSink_b + ax_R8",
        "Uniform_k16 + MeanSink_b + ax_R16",
        "Oracle_AttnKV + MeanSink_b",
        "Oracle_AttnKV + MeanSink_b + ax_R8",
        "Oracle_AttnKV + MeanSink_b + ax_R16",
    )
    oracle_summary = tuple(
        _finalize_policy_metrics(name, oracle_metrics[name], oracle_schedules[name])
        for name in oracle_order
        if oracle_metrics[name]["all_asst"].count > 0
    )
    state_err_summaries = {
        name: _finalize_state_error_summaries(name, state_err_acc[name])
        for name in state_err_targets
    }
    return prefill_summary, oracle_summary, state_err_summaries, trace_meta
