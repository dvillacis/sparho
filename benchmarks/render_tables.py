#!/usr/bin/env python
"""Render benchmark results JSON into Markdown tables.

Reads one or more ``--results`` JSON files produced by
``benchmarks/lasso_libsvm.py --results-json …`` and emits the
canonical Markdown tables embedded in ``benchmarks/README.md`` and
the top-level ``README.md``.

Usage::

    uv run python benchmarks/render_tables.py \\
        --results /tmp/bench-results.json \\
        --provenance /tmp/bench-provenance.json

By default writes to stdout. Pass ``--out PATH`` to write to a file.

The renderer is deterministic and pure — re-running it with the same
JSON produces byte-identical Markdown. That makes the README tables
*derived artifacts* rather than hand-maintained prose, which fixes the
long-standing drift problem where the README claims one number and the
code measures another.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _format_alpha(alpha: float) -> str:
    """Compact alpha: 4 sig figs, scientific for very small."""
    if abs(alpha) < 1e-3:
        return f"{alpha:.2e}"
    return f"{alpha:.4g}"


def _format_time(t: float) -> str:
    if t < 1.0:
        return f"{t:.2f} s"
    if t < 100.0:
        return f"{t:.2f} s"
    return f"{t:.0f} s"


def _format_speedup(s: float) -> str:
    if s >= 1.0:
        return f"**{s:.2f}×**"
    return f"{s:.2f}×"


def _format_shape(shape: list[int], sparse: bool) -> str:
    base = f"{shape[0]}×{shape[1]}"
    return f"{base} sparse" if sparse else base


def render_speedup_table(results: list[dict[str, Any]]) -> str:
    """Render the headline 'sparho vs LassoCV' speedup table."""
    header = (
        "| dataset | shape "
        "| sparho α* | sparho MSE | sparho time | sparho iters "
        "| LassoCV α* | LassoCV MSE | LassoCV time | LassoCV grid "
        "| speedup |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = []
    for r in results:
        s = r["sparho"]
        lcv = r["lasso_cv"]
        rows.append(
            f"| `{r['dataset']}` | {_format_shape(r['shape'], r['sparse'])} "
            f"| {_format_alpha(s['alpha_star'])} | {s['mse']:.4g} "
            f"| {_format_time(s['elapsed_seconds'])} | {s['iters']} "
            f"| {_format_alpha(lcv['alpha_star'])} | {lcv['mse']:.4g} "
            f"| {_format_time(lcv['elapsed_seconds'])} | {lcv['grid_size']} "
            f"| {_format_speedup(r['speedup'])} |"
        )
    return "\n".join([header, sep, *rows])


def render_spread_table(results: list[dict[str, Any]]) -> str:
    """Render the reproducibility-spread table — only meaningful for --repeat > 1."""
    header = "| dataset | sparho spread | LassoCV spread |"
    sep = "|---|---|---|"
    rows = []
    for r in results:
        s_n = r["sparho"]["n_samples"]
        l_n = r["lasso_cv"]["n_samples"]
        if s_n <= 1 and l_n <= 1:
            continue  # nothing to report from a single-run
        rows.append(
            f"| `{r['dataset']}` {_format_shape(r['shape'], r['sparse'])} "
            f"| ±{100 * r['sparho']['elapsed_spread']:.1f}% "
            f"| ±{100 * r['lasso_cv']['elapsed_spread']:.1f}% |"
        )
    if not rows:
        return "_no multi-sample data — re-run with `--repeat 5` to populate this table._"
    return "\n".join([header, sep, *rows])


def render_provenance_summary(provenance: dict[str, Any]) -> str:
    """Compact one-paragraph provenance attribution."""
    p = provenance.get("platform", {})
    v = provenance.get("package_versions", {})
    threads = provenance.get("blas_threads_requested", "?")
    git = provenance.get("git_sha", "unknown")
    git_short = git[:8] if isinstance(git, str) and git != "unknown" else git
    return (
        f"_Measured at git `{git_short}` on "
        f"{p.get('system', '?')} {p.get('machine', '?')} "
        f"({p.get('processor', '?')}), Python {p.get('python_version', '?')}, "
        f"numpy {v.get('numpy', '?')}, scipy {v.get('scipy', '?')}, "
        f"scikit-learn {v.get('scikit-learn', '?')}, "
        f"celer {v.get('celer', 'n/a')}, BLAS threads = {threads}._"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="path to results JSON")
    parser.add_argument("--provenance", type=Path, default=None, help="optional provenance JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write rendered Markdown here (default: stdout)",
    )
    parser.add_argument(
        "--include",
        choices=("speedup", "spread", "all"),
        default="all",
        help="which table(s) to render",
    )
    args = parser.parse_args()

    blob = json.loads(args.results.read_text())
    results = blob["datasets"]

    sections: list[str] = []
    if args.include in ("speedup", "all"):
        sections.append("## Speedup vs LassoCV\n\n" + render_speedup_table(results))
    if args.include in ("spread", "all"):
        sections.append("## Reproducibility spread\n\n" + render_spread_table(results))
    if args.provenance is not None:
        provenance = json.loads(args.provenance.read_text())
        sections.append(render_provenance_summary(provenance))

    rendered = "\n\n".join(sections) + "\n"
    if args.out is None:
        sys.stdout.write(rendered)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
