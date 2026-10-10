"""Markdown report and figures from ``results/<study>/evaluation.json``.

The report follows the pre-registered plan (``docs/preregistration.md``): the sample, the primary
hypotheses (H1 encompassing, H2 out-of-sample gain over ``market_recal``), calibration, secondary
analyses and P&L after fees. It states numbers, not conclusions.

The primary family is re-adjusted here across every horizon x forecaster x condition, as
pre-registered. ``evaluate`` runs one forecaster key at a time, so the ``primary`` block in each
key's results is Holm-adjusted over that key's horizons only.

Formats: Brier and its differences to 4 decimals; Delta = loss(benchmark) - loss(model), so a
positive Delta means the model is better; p-values to 3 decimals; money in dollars.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pricedin.eval.inference import holm
from pricedin.plots import HORIZON_ORDER, short_key
from pricedin.schema import read_jsonl
from pricedin.universe import Study

MIN_CELL = 30  # breakdown cells below this are flagged (pre-registered)


# --- formatting ---------------------------------------------------------------------------------


def _ok(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def fb(x: Any) -> str:
    """Brier score or a Brier difference: 4 decimals."""
    return f"{x:.4f}" if _ok(x) else "–"


def fd(x: Any) -> str:
    """Signed difference: 4 decimals."""
    return f"{x:+.4f}" if _ok(x) else "–"


def fci(d: dict[str, Any] | None, f=fd) -> str:
    """``+0.0123 [-0.0010, +0.0250]``."""
    if not d:
        return "–"
    m = d.get("mean", d.get("estimate"))
    lo, hi = d.get("lo"), d.get("hi")
    if _ok(lo) and _ok(hi):
        return f"{f(m)} [{f(lo)}, {f(hi)}]"
    return f(m)


def fp(p: Any) -> str:
    if not _ok(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fpct(x: Any) -> str:
    return f"{100 * x:.1f}%" if _ok(x) else "–"


def fusd(x: Any) -> str:
    return f"{x:+,.0f}" if _ok(x) else "–"


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    """GitHub Markdown table; ``align`` is one char per column (l/r), default l then r."""
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# --- reading ------------------------------------------------------------------------------------


def load_evaluation(out_dir: str | Path) -> dict[str, Any]:
    p = Path(out_dir) / "evaluation.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `pricedin evaluate` first")
    return json.loads(p.read_text(encoding="utf-8"))


def _horizons(res: dict[str, Any]) -> list[str]:
    return sorted(
        res["meta"]["horizons"],
        key=lambda h: HORIZON_ORDER.index(h) if h in HORIZON_ORDER else 9,
    )


def _fblock(res: dict[str, Any], key: str, h: str) -> dict[str, Any]:
    return res.get("by_horizon", {}).get(h, {}).get("forecasters", {}).get(key, {})


def _model(res: dict[str, Any], key: str, h: str, name: str) -> dict[str, Any]:
    return _fblock(res, key, h).get("models", {}).get(name, {})


def _delta(m: dict[str, Any], bench: str = "market_recal", loss: str = "brier") -> dict | None:
    return m.get(f"delta_vs_{bench}", {}).get(loss)


def _enc(res: dict[str, Any], key: str, h: str) -> dict[str, Any]:
    return _fblock(res, key, h).get("encompassing", {}).get("full", {})


def primary_family(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    """H1 and H2 for every forecaster|condition x horizon (pooled "all" excluded), each
    Holm-adjusted across the whole family. H2 rejects only when the gain is positive."""
    alpha = 0.05
    rows = []
    for key, res in evaluation["keys"].items():
        alpha = res["meta"].get("alpha", alpha)
        for h in _horizons(res):
            pool = _model(res, key, h, "pool")
            d = _delta(pool) or {}
            e = _enc(res, key, h)
            rows.append(
                {
                    "key": key,
                    "horizon": h,
                    "n_test": _fblock(res, key, h).get("n_test"),
                    "delta": d.get("mean"),
                    "lo": d.get("lo"),
                    "hi": d.get("hi"),
                    "p_h2": d.get("p"),
                    "c": e.get("coef", {}).get("llm"),
                    "c_z": e.get("llm_adds_info", {}).get("z"),
                    "p_h1": e.get("llm_adds_info", {}).get("p"),
                    "b": e.get("coef", {}).get("market"),
                }
            )
    for test in ("h1", "h2"):
        p = np.array([r[f"p_{test}"] if _ok(r[f"p_{test}"]) else np.nan for r in rows], float)
        adj = holm(p) if len(p) else p
        for r, a in zip(rows, adj, strict=True):
            r[f"p_{test}_holm"] = None if np.isnan(a) else float(a)
    for r in rows:
        r["reject_h2"] = bool(
            r["p_h2_holm"] is not None and r["p_h2_holm"] < alpha and (r["delta"] or 0) > 0
        )
        r["reject_h1"] = bool(r["p_h1_holm"] is not None and r["p_h1_holm"] < alpha)
    return rows


def featured_key(evaluation: dict[str, Any]) -> str:
    """The forecaster|condition with the lowest test-period LLM Brier, pooled over horizons
    (or the single horizon). Shown in the single-forecaster figures."""
    best, best_b = None, math.inf
    for key, res in evaluation["keys"].items():
        hs = _horizons(res)
        h = "all" if len(hs) > 1 else hs[0]
        b = _model(res, key, h, "llm").get("brier")
        if _ok(b) and b < best_b:
            best, best_b = key, b
    return best or next(iter(evaluation["keys"]))


def summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Compact headline numbers (JSON-safe) for the README."""
    fam = {(r["key"], r["horizon"]): r for r in primary_family(evaluation)}
    out: dict[str, Any] = {
        "study": evaluation.get("study"),
        "split_at": evaluation.get("split_at"),
        "featured": featured_key(evaluation),
        "delta": "Brier(market_recal) - Brier(pool); positive = pool better",
        "keys": {},
    }
    for key, res in evaluation["keys"].items():
        hs = _horizons(res)
        k: dict[str, Any] = {}
        for h in [*hs, *(["all"] if len(hs) > 1 else [])]:
            ms = {
                m: _model(res, key, h, m).get("brier")
                for m in ("market", "market_recal", "llm", "pool")
            }
            d = _delta(_model(res, key, h, "pool")) or {}
            e = _enc(res, key, h)
            row = {
                "n_test": _fblock(res, key, h).get("n_test"),
                "brier": ms,
                "delta": d.get("mean"),
                "delta_ci": [d.get("lo"), d.get("hi")],
                "c": e.get("coef", {}).get("llm"),
                "c_z": e.get("llm_adds_info", {}).get("z"),
            }
            if (key, h) in fam:
                f = fam[(key, h)]
                row |= {
                    "p_h1_holm": f["p_h1_holm"],
                    "p_h2_holm": f["p_h2_holm"],
                    "reject_h1": f["reject_h1"],
                    "reject_h2": f["reject_h2"],
                }
            k[h] = row
        pnl = res.get("pnl", {}).get("by_group", {}).get("all", {})
        if pnl:
            k["pnl"] = {
                m: {
                    "pnl": v.get("pnl"),
                    "trades": v.get("n_trades"),
                    "roi": v.get("roi", {}).get("estimate"),
                }
                for m, v in pnl.get("models", {}).items()
            }
            inc = pnl.get("incremental_vs_market_recal", {}).get("pool")
            if inc:
                k["pnl"]["pool_incremental_per_row"] = {
                    "mean": inc.get("mean"),
                    "ci": [inc.get("lo"), inc.get("hi")],
                }
        out["keys"][key] = k
    return out


# --- sections -----------------------------------------------------------------------------------


def _sample_section(study: Study, ev: dict[str, Any]) -> list[str]:
    out = ["## Sample", ""]
    sp = study.data_dir / "sample.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        out.append(
            f"{s['candidates']} eligible candidates in {s['events']} events. "
            f"{s['after_event_cap']} remain after the per-event cap ({study.per_event}). "
            f"{s['sampled']} are sampled after the per-cell cap ({study.per_cell})."
        )
        out.append("")
    qs = list(read_jsonl(study.questions_path)) if study.questions_path.exists() else []
    if qs:
        venues = sorted({q["venue"] for q in qs})
        cells = Counter((q["category"], q["venue"]) for q in qs)
        cats = sorted({c for c, _ in cells})
        rows = [
            [c, *[str(cells[(c, v)]) for v in venues], str(sum(cells[(c, v)] for v in venues))]
            for c in cats
        ]
        rows.append(
            [
                "**total**",
                *[str(sum(cells[(c, v)] for c in cats)) for v in venues],
                f"**{len(qs)}**",
            ]
        )
        out += [
            "Questions by category and venue:",
            "",
            table(["category", *venues, "total"], rows),
            "",
        ]
        early = sum(bool(q.get("extra", {}).get("early_close")) for q in qs)
        out += [f"{early} questions ({fpct(early / len(qs))}) closed early.", ""]

    snaps = (
        Counter(r["horizon"] for r in read_jsonl(study.snapshots_path))
        if study.snapshots_path.exists()
        else Counter()
    )
    mp = study.data_dir / "snapshots_missing.json"
    missing = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    miss = Counter(k.rsplit("|", 1)[-1] for k in missing)
    keys = list(ev["keys"])
    hs = [h for h in HORIZON_ORDER if h in study.horizons] + [
        h for h in study.horizons if h not in HORIZON_ORDER
    ]
    header = ["horizon", "snapshots", "missing"] + [f"rows: {short_key(k)}" for k in keys]
    rows = []
    for h in hs:
        r = [h, str(snaps.get(h, 0)), str(miss.get(h, 0))]
        r += [str(ev["keys"][k]["by_horizon"].get(h, {}).get("n", 0)) for k in keys]
        rows.append(r)
    out += [
        "Rows per horizon. A row needs a snapshot and a parsed forecast.",
        "",
        table(header, rows),
        "",
    ]
    drops = []
    for k in keys:
        d = ev["keys"][k]["meta"].get("panel_dropped", {})
        if d:
            parts = ", ".join(f"{n} {r.split(':')[0].replace('_', ' ')}" for r, n in d.items())
            drops.append(f"- {short_key(k)}: {parts}.")
    if drops:
        out += ["Rows dropped while building each panel (question × horizon):", "", *drops, ""]
    if missing:
        reasons = Counter(v.split(":")[0] for v in missing.values())
        out += [
            "Missing snapshots by reason: "
            + "; ".join(f"{n} × {r}" for r, n in reasons.most_common())
            + ".",
            "",
        ]

    out += [
        f"Split at **{ev.get('split_at')}**. Training rows closed before it. Test rows were "
        "snapshotted at or after it. Rows that straddle it are embargoed.",
        "",
    ]
    rows = []
    for k in keys:
        res = ev["keys"][k]
        for h in _horizons(res):
            b = res["by_horizon"][h]
            rows.append(
                [short_key(k), h, str(b["n_train"]), str(b["n_test"]), str(b["n_embargoed"])]
            )
    out += [table(["forecaster", "h", "train", "test", "embargoed"], rows, "llrrr"), ""]
    return out


def _headline_section(ev: dict[str, Any]) -> list[str]:
    fam = primary_family(ev)
    out = [
        "## Primary hypotheses",
        "",
        "Test-period Brier scores (lower is better). "
        "Δ = Brier(market_recal) − Brier(pool), positive when adding the LLM helps; "
        "95% CI from an event-clustered bootstrap. "
        "c is the LLM coefficient in logit(y) = a + b·logit(market) + c·logit(LLM), "
        "fitted on the full sample with event-clustered (CR1) errors. "
        f"Holm adjustment is over all {len(fam)} forecaster × horizon tests, "
        "separately for H1 and H2.",
        "",
    ]
    rows = []
    for r in fam:
        res = ev["keys"][r["key"]]
        b = {
            m: _model(res, r["key"], r["horizon"], m).get("brier")
            for m in ("market", "market_recal", "llm", "pool")
        }
        rows.append(
            [
                short_key(r["key"]),
                r["horizon"],
                str(r["n_test"] if r["n_test"] is not None else "–"),
                fb(b["market"]),
                fb(b["market_recal"]),
                fb(b["llm"]),
                fb(b["pool"]),
                fci({"mean": r["delta"], "lo": r["lo"], "hi": r["hi"]}),
                fp(r["p_h2_holm"]) + (" *" if r["reject_h2"] else ""),
                f"{r['c']:+.2f} (z {r['c_z']:+.1f})" if _ok(r["c"]) and _ok(r["c_z"]) else "–",
                fp(r["p_h1_holm"]) + (" *" if r["reject_h1"] else ""),
            ]
        )
    out += [
        table(
            [
                "forecaster",
                "h",
                "n test",
                "market",
                "mkt recal",
                "LLM",
                "pool",
                "Δ [95% CI]",
                "H2 p (Holm)",
                "c (z)",
                "H1 p (Holm)",
            ],
            rows,
            "llrrrrrrrrr",
        ),
        "",
        "\\* rejects at α = 0.05 after Holm adjustment. H2 counts only gains (Δ > 0).",
        "",
    ]
    pooled = []
    for key, res in ev["keys"].items():
        if "all" in res["by_horizon"]:
            pool = _model(res, key, "all", "pool")
            n = _fblock(res, key, "all").get("n_test", "–")
            pooled.append(
                f"- {short_key(key)}: Δ {fci(_delta(pool))} on {n} "
                "test rows (all horizons pooled; not part of the family)."
            )
    if pooled:
        out += pooled + [""]
    out += ["![Δ Brier by forecaster and horizon](figures/forest.png)", ""]
    out += ["![Encompassing coefficients](figures/encompassing.png)", ""]
    return out


def _calibration_section(ev: dict[str, Any], featured: str) -> list[str]:
    out = [
        "## Calibration",
        "",
        "Expected calibration error (ECE, equal-mass bins) and the Murphy decomposition "
        "Brier ≈ REL − RES + UNC on the test period. Lower REL is better calibrated; higher "
        "RES separates outcomes better.",
        "",
    ]
    rows = []
    for key, res in ev["keys"].items():
        for h in _horizons(res):
            r = [short_key(key), h]
            for m in ("market", "llm", "pool"):
                mm = _model(res, key, h, m)
                r.append(fb(mm.get("ece")))
            for m in ("market", "llm", "pool"):
                mu = _model(res, key, h, m).get("murphy", {})
                r.append(f"{fb(mu.get('reliability'))} / {fb(mu.get('resolution'))}")
            unc = _model(res, key, h, "market").get("murphy", {}).get("uncertainty")
            r.append(fb(unc))
            rows.append(r)
    out += [
        table(
            [
                "forecaster",
                "h",
                "ECE market",
                "ECE LLM",
                "ECE pool",
                "REL/RES market",
                "REL/RES LLM",
                "REL/RES pool",
                "UNC",
            ],
            rows,
            "llrrrrrrr",
        ),
        "",
        "![Brier by horizon](figures/brier_by_horizon.png)",
        "",
        f"![Reliability diagrams: {short_key(featured)}](figures/reliability.png)",
        "",
    ]
    return out


def _subset_rows(res: dict[str, Any], key: str, name: str) -> list[list[str]]:
    sub = res.get("subsets", {}).get(name)
    if sub is None:
        return []
    if "error" in sub:
        return [[short_key(key), "–", "–", f"error: {sub['error']}", "–"]]
    rows = []
    for h in _horizons(sub):
        fbk = _fblock(sub, key, h)
        if "skipped" in fbk:
            rows.append(
                [short_key(key), h, str(fbk.get("n_test", "–")), f"skipped: {fbk['skipped']}", "–"]
            )
            continue
        d = _delta(_model(sub, key, h, "pool"))
        rows.append(
            [short_key(key), h, str(fbk.get("n_test", "–")), fci(d), fp((d or {}).get("p"))]
        )
    return rows


def _secondary_section(ev: dict[str, Any]) -> list[str]:
    out = ["## Secondary analyses", ""]
    keys = list(ev["keys"])

    out += [
        "### Other contrasts and ablations",
        "",
        "Test period. Δ columns are positive when the pool is better. The last three columns are "
        "Brier scores of the LLM-only variants and the equal-weight logit average.",
        "",
    ]
    rows = []
    for key in keys:
        res = ev["keys"][key]
        for h in _horizons(res):
            pool = _model(res, key, h, "pool")
            rows.append(
                [
                    short_key(key),
                    h,
                    fci(_delta(pool, "market")),
                    fci(_delta(pool, "market_recal", "log_loss")),
                    fb(_model(res, key, h, "llm_recal").get("brier")),
                    fb(_model(res, key, h, "llm_extremized").get("brier")),
                    fb(_model(res, key, h, "average").get("brier")),
                ]
            )
    out += [
        table(
            [
                "forecaster",
                "h",
                "Δ Brier vs raw market",
                "Δ log score vs mkt recal",
                "LLM recal",
                "LLM extremized",
                "average",
            ],
            rows,
            "llrrrrr",
        ),
        "",
    ]

    # news vs base: the retrieval ablation, when both conditions ran for one forecaster
    pairs = sorted(
        {k.partition("|")[0] for k in keys if k.endswith("|news")}
        & {k.partition("|")[0] for k in keys if k.endswith("|base")}
    )
    if pairs:
        out += [
            "### News vs base",
            "",
            "Test-period Brier of the LLM alone and Δ of the pool, per condition. The two "
            "conditions can cover slightly different rows; no paired CI.",
            "",
        ]
        rows = []
        for f in pairs:
            b, n = ev["keys"][f"{f}|base"], ev["keys"][f"{f}|news"]
            for h in _horizons(b):
                rows.append(
                    [
                        short_key(f),
                        h,
                        fb(_model(b, f"{f}|base", h, "llm").get("brier")),
                        fb(_model(n, f"{f}|news", h, "llm").get("brier")),
                        fd((_delta(_model(b, f"{f}|base", h, "pool")) or {}).get("mean")),
                        fd((_delta(_model(n, f"{f}|news", h, "pool")) or {}).get("mean")),
                    ]
                )
        out += [
            table(
                ["forecaster", "h", "LLM base", "LLM news", "Δ pool base", "Δ pool news"],
                rows,
                "llrrrr",
            ),
            "",
        ]

    for name, title, desc in (
        (
            "contested",
            "Contested markets",
            "Rows whose market mid was inside the contested band at the snapshot.",
        ),
        ("no_early_close", "No early close", "Questions whose market did not close early."),
    ):
        rows = [r for k in keys for r in _subset_rows(ev["keys"][k], k, name)]
        if rows:
            out += [
                f"### {title}",
                "",
                desc + " Same split; models refitted on the subset.",
                "",
                table(
                    ["forecaster", "h", "n test", "Δ pool vs mkt recal [95% CI]", "p"],
                    rows,
                    "llrrr",
                ),
                "",
            ]

    for dim in ("category", "venue"):
        rows = []
        for key in keys:
            res = ev["keys"][key]
            hs = _horizons(res)
            h = "all" if "all" in res["by_horizon"] else hs[0]
            for g, cell in _fblock(res, key, h).get("breakdowns", {}).get(dim, {}).items():
                d = cell.get("delta_pool_vs_market_recal")
                rows.append(
                    [
                        short_key(key),
                        g,
                        str(cell["n"]) + (" †" if cell.get("small") else ""),
                        str(cell.get("n_clusters", "–")),
                        fb(cell["brier"].get("market_recal")),
                        fb(cell["brier"].get("pool")),
                        fci(d),
                    ]
                )
        if rows:
            out += [
                f"### By {dim}",
                "",
                "Test period, all horizons pooled. † fewer than "
                f"{MIN_CELL} rows: read as descriptive only.",
                "",
                table(
                    ["forecaster", dim, "n", "events", "mkt recal", "pool", "Δ [95% CI]"],
                    rows,
                    "llrrrrr",
                ),
                "",
            ]

    rows = []
    for key in keys:
        wf = ev["keys"][key].get("walk_forward", {})
        for h, blocks in wf.items():
            w = blocks.get(key, {})
            if "skipped" in w:
                rows.append([short_key(key), h, "0", "–", "–", "–", f"skipped: {w['skipped']}"])
                continue
            ms = w.get("models", {})
            d = ms.get("pool", {}).get("delta_vs_market_recal", {}).get("brier")
            rows.append(
                [
                    short_key(key),
                    h,
                    str(w.get("n_folds", 0)),
                    str(w.get("n_test", "–")),
                    fb(ms.get("market_recal", {}).get("brier")),
                    fb(ms.get("pool", {}).get("brier")),
                    fci(d),
                ]
            )
    if rows:
        out += [
            "### Walk-forward",
            "",
            "Monthly expanding-window refits; out-of-fold predictions scored together.",
            "",
            table(
                ["forecaster", "h", "folds", "n test", "mkt recal", "pool", "Δ [95% CI]"],
                rows,
                "llrrrrr",
            ),
            "",
        ]
    return out


def _pnl_section(ev: dict[str, Any], featured: str) -> list[str]:
    out = ["## P&L after fees", ""]
    meta = next((r["pnl"]["meta"] for r in ev["keys"].values() if "meta" in r.get("pnl", {})), None)
    if meta:
        out += [
            f"Each model buys {meta['sizing'].get('contracts')} contracts as a taker when its "
            f"expected edge after fees exceeds {meta['threshold'] * 100:.0f}¢, and holds to "
            "resolution. Kalshi rows trade at the candle bid/ask with the quadratic taker fee. "
            "Polymarket rows have no historical bid/ask: they trade at an assumed spread around "
            "the mid and are **approximate**. ROI = net P&L / (cost + fees); CIs are "
            "event-clustered bootstraps.",
            "",
        ]
    for key, res in ev["keys"].items():
        pnl = res.get("pnl", {})
        if "error" in pnl:
            out += [f"{short_key(key)}: P&L not computed ({pnl['error']}).", ""]
            continue
        groups = pnl.get("by_group", {})
        if not groups:
            continue
        out += [f"### {short_key(key)}", ""]
        rows = []
        for g in ("all", *[g for g in groups if g.startswith("venue:")]):
            if g not in groups:
                continue
            for m in ("market_recal", "pool", "llm"):
                s = groups[g]["models"].get(m)
                if not s:
                    continue
                label = g.removeprefix("venue:") + (" (approx.)" if g == "venue:polymarket" else "")
                approx = s.get("n_approximate", 0)
                rows.append(
                    [
                        label,
                        m,
                        f"{s['n_trades']}"
                        + (f" ({approx} approx.)" if approx and g == "all" else ""),
                        fpct(s.get("hit_rate")),
                        fusd(s.get("pnl")),
                        fci(s.get("roi"), fpct),
                        f"{s.get('fees', 0):,.0f}",
                    ]
                )
        out += [
            table(
                ["rows", "model", "trades", "hit rate", "net P&L ($)", "ROI [95% CI]", "fees ($)"],
                rows,
                "llrrrrr",
            ),
            "",
        ]
        inc = groups.get("all", {}).get("incremental_vs_market_recal", {})
        if inc:
            parts = [
                f"{m} {fci(d, lambda x: f'{x:+.2f}')} (p {fp(d.get('p'))})"
                for m, d in inc.items()
                if m in ("pool", "llm")
            ]
            out += [
                "Incremental P&L per test row vs market_recal ($, 95% CI): "
                + "; ".join(parts)
                + ".",
                "",
            ]
    out += [f"![Cumulative P&L: {short_key(featured)}](figures/pnl.png)", ""]
    return out


# --- entry point --------------------------------------------------------------------------------


def render(study: Study, evaluation: dict[str, Any], figures: bool = True) -> str:
    """The report as Markdown (figures are referenced under ``figures/``)."""
    featured = featured_key(evaluation)
    lines = [
        f"# pricedin report: study {evaluation.get('study', study.name)}",
        "",
        f"Generated by pricedin {evaluation.get('version', '')} from `evaluation.json`. "
        "The analysis plan is in `docs/preregistration.md`. "
        f"Figures for a single forecaster show **{short_key(featured)}**, the lowest LLM Brier.",
        "",
    ]
    if not figures:
        lines += ["Figures were not drawn (matplotlib not installed).", ""]
    lines += _sample_section(study, evaluation)
    lines += _headline_section(evaluation)
    lines += _calibration_section(evaluation, featured)
    lines += _secondary_section(evaluation)
    lines += _pnl_section(evaluation, featured)
    text = "\n".join(lines).rstrip() + "\n"
    if not figures:
        text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("![")) + "\n"
    return text


def write_report(study: Study, out_dir: str | Path, *, figures: bool = True) -> Path:
    """Read ``out_dir/evaluation.json``; write ``out_dir/report.md`` and ``out_dir/figures/``.

    Figures need matplotlib; without it the report is written without them and a note says so.
    Returns the report path.
    """
    out_dir = Path(out_dir)
    evaluation = load_evaluation(out_dir)
    drew = False
    if figures:
        try:
            from pricedin import plots

            plots.all_figures(evaluation, out_dir / "figures", featured_key(evaluation))
            drew = True
        except ImportError:
            drew = False
    path = out_dir / "report.md"
    path.write_text(render(study, evaluation, figures=drew), encoding="utf-8", newline="\n")
    (out_dir / "summary.json").write_text(
        json.dumps(summary(evaluation), indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return path
