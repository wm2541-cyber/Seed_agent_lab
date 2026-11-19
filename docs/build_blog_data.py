"""Pure script to bundle verified JSON benchmark artifacts into `docs/blog_data.js` for the interactive web blog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(__file__).resolve().parent


def _load_json(rel_path: str) -> Any:
    return json.loads((REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def build_blog_data_payload() -> dict[str, Any]:
    """Purely assemble compact, chart-ready data from the 4 verified benchmark JSON reports."""
    e2e = _load_json("unified_e2e_pipeline/artifacts/unified_e2e_results.json")
    d1 = _load_json("direction1_phase_shortlist/artifacts/benchmark_results.json")
    d2 = _load_json("direction2_hybrid_prefill_align/artifacts/direction2_eval_results.json")
    d3 = _load_json("direction3_role_prefill_svd/artifacts/direction3_svd_report.json")

    # Slim down per-layer weight SVD to essential plotting fields so blog_data.js stays lightweight (<50KB)
    weight_layers = []
    for rec in d3["weight_svd_layers"]:
        proj_map = {name: spec for name, spec in rec["proj_spectra"]}
        entry: dict[str, Any] = {
            "layer_idx": rec["layer_idx"],
            "layer_type": rec["layer_type"],
            "eff_ranks": {k: round(float(v["effective_rank"]), 1) for k, v in proj_map.items()},
            "r90": {k: int(v["r90"]) for k, v in proj_map.items()},
        }
        if rec.get("mean_sink_top1_cos") is not None:
            entry["sink_top1_cos"] = round(float(rec["mean_sink_top1_cos"]), 3)
            entry["sink_top4_energy"] = round(float(rec["mean_sink_top4_energy"]) * 100.0, 1)
            entry["wout_sink_gain"] = round(float(rec["mean_wout_sink_gain"]), 3)
        weight_layers.append(entry)

    return {
        "e2e": {
            "aggregates": e2e["aggregates"],
            "per_prompt": e2e["per_prompt"],
        },
        "d1": {
            "logz_held_out": d1["logz_held_out"],
            "offline_shortlist_diagnostics": d1["offline_shortlist_diagnostics"],
            "summary": d1["summary"],
            "per_prompt_records": d1["per_prompt_records"],
        },
        "d2": {
            "total_prefix_tokens": d2["total_prefix_tokens"],
            "summaries": d2["summaries"],
        },
        "d3": {
            "trace_meta": d3["trace_meta"],
            "prefill_summary": d3["prefill_summary"],
            "oracle_summary": d3["oracle_summary"],
            "weight_svd_by_depth": d3["weight_svd_by_depth"],
            "weight_layers": weight_layers,
            "state_error_u16": d3["state_error_svd"].get("Uniform_k16", []),
        },
    }


def main() -> None:
    payload = build_blog_data_payload()
    out_path = DOCS_DIR / "blog_data.js"
    js_content = (
        "// Auto-generated from verified benchmark JSON artifacts by docs/build_blog_data.py\n"
        f"window.SEED_BLOG_DATA = {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};\n"
    )
    out_path.write_text(js_content, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
