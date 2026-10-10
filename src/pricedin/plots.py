"""Figures for the report, drawn from ``evaluation.json`` with matplotlib (optional dependency).

Every figure is saved as PNG (for the README) and SVG, on a light surface with recessive axes.
Colour follows the model, never its rank: ``market`` grey (the raw reference), ``market_recal``
blue, ``llm`` aqua, ``pool`` orange; the three hues pass the CVD and normal-vision checks
all-pairs (dataviz validator, light mode). Aqua is below 3:1 on the surface, so every multi-series
figure carries a legend or direct labels. Output is deterministic: fixed SVG hash salt, no date
metadata, no timestamps.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

HORIZON_ORDER = ("7d", "1d", "1h")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLORS = {
    "market": MUTED,
    "market_recal": "#2a78d6",
    "llm": "#1baf7a",
    "pool": "#eb6834",
}
LABELS = {
    "market": "market",
    "market_recal": "market, recalibrated",
    "llm": "LLM",
    "pool": "pool (market + LLM)",
}
DPI = 150
LW = 1.5  # points; ~2 px lines at 150 dpi after anti-aliasing
MS = 5.5  # marker size in points (>= 8 px)


def _mpl():
    try:
        import matplotlib
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "figures need matplotlib: install with `uv sync --extra plot` "
            "or `pip install 'pricedin[plot]'`"
        ) from exc
    matplotlib.rcParams.update(
        {
            "svg.hashsalt": "pricedin",
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.facecolor": SURFACE,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.facecolor": SURFACE,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )
    return Figure


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png, svg = out_dir / f"{name}.png", out_dir / f"{name}.svg"
    fig.savefig(png, dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
    fig.savefig(svg, facecolor=SURFACE, metadata={"Date": None, "Creator": None})
    return [png, svg]


def _legend_below(fig, ax, ncol: int | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=ncol or len(handles))


def short_key(key: str) -> str:
    """``claude-cli:claude-opus-5-5@low|base`` -> ``claude-opus-5-5@low · base``."""
    forecaster, _, condition = key.partition("|")
    name = forecaster.split(":", 1)[-1]
    return f"{name} · {condition}" if condition else name


def _horizons(res: dict[str, Any]) -> list[str]:
    hs = res["meta"]["horizons"]
    return sorted(hs, key=lambda h: HORIZON_ORDER.index(h) if h in HORIZON_ORDER else 9)


def _pool_block(res: dict[str, Any], key: str, h: str) -> dict[str, Any] | None:
    return res["by_horizon"].get(h, {}).get("forecasters", {}).get(key, {}).get("models")


# --- figures ------------------------------------------------------------------------------------


def forest(evaluation: dict[str, Any], out_dir: Path, name: str = "forest") -> list[Path]:
    """Delta Brier (market_recal - pool; right of zero = pool better) with 95% CI, one row per
    forecaster|condition x horizon."""
    Figure = _mpl()
    rows = []
    for key, res in evaluation["keys"].items():
        for h in _horizons(res):
            models = _pool_block(res, key, h)
            if not models or "pool" not in models:
                continue
            d = models["pool"]["delta_vs_market_recal"]["brier"]
            rows.append((f"{short_key(key)}  {h}", d["mean"], d.get("lo"), d.get("hi")))
    fig = Figure(figsize=(6.4, 0.9 + 0.32 * max(len(rows), 1)))
    ax = fig.add_subplot()
    ys = list(range(len(rows)))[::-1]
    for y, (_, m, lo, hi) in zip(ys, rows, strict=True):
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y, y], color=COLORS["pool"], lw=LW)
        ax.plot([m], [y], "o", color=COLORS["pool"], ms=MS, mec=SURFACE, mew=1.2)
    ax.axvline(0, color=INK_2, lw=0.9)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Δ Brier = market_recal − pool  (right of 0: pool better)")
    ax.set_title("Out-of-sample gain from adding the LLM, 95% CI", loc="left")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def reliability(
    evaluation: dict[str, Any], key: str, out_dir: Path, name: str = "reliability"
) -> list[Path]:
    """Reliability diagrams per horizon (equal-mass bins) for market, LLM and pool on the test
    period of ``key``."""
    Figure = _mpl()
    res = evaluation["keys"][key]
    hs = [h for h in _horizons(res) if _pool_block(res, key, h)]
    fig = Figure(figsize=(2.6 * max(len(hs), 1) + 0.4, 3.7))
    axes = fig.subplots(1, max(len(hs), 1), sharey=True, squeeze=False)[0]
    for ax, h in zip(axes, hs, strict=False):
        models = _pool_block(res, key, h)
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.9)
        n = 0
        for m in ("market", "llm", "pool"):
            rb = models.get(m, {}).get("reliability")
            if not rb:
                continue
            n = sum(rb["count"])
            ax.plot(
                rb["mean_p"],
                rb["freq"],
                "-o",
                color=COLORS[m],
                lw=LW,
                ms=MS - 1,
                mec=SURFACE,
                mew=1,
                label=LABELS[m],
            )
        k = len(models.get("pool", {}).get("reliability", {}).get("count", [])) or 1
        ax.set_title(f"{h}  (n = {n}, ~{round(n / k)} per bin)", loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("mean forecast in bin")
    axes[0].set_ylabel("observed frequency")
    _legend_below(fig, axes[0])
    fig.suptitle(
        f"Calibration on the test period: {short_key(key)}  (equal-mass bins)",
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return _save(fig, out_dir, name)


def encompassing(
    evaluation: dict[str, Any], out_dir: Path, name: str = "encompassing", z: float = 1.96
) -> list[Path]:
    """Encompassing coefficients b (market) and c (LLM) by horizon, full sample, with
    cluster-robust 95% CIs; one panel per forecaster|condition."""
    Figure = _mpl()
    keys = list(evaluation["keys"])
    ncol = min(len(keys), 3) or 1
    nrow = -(-len(keys) // ncol) or 1
    fig = Figure(figsize=(3.0 * ncol + 0.4, 2.7 * nrow + 0.3))
    axes = fig.subplots(nrow, ncol, sharey=True, squeeze=False).ravel()
    for ax in axes[len(keys) :]:
        ax.set_visible(False)
    for ax, key in zip(axes, keys, strict=False):
        res = evaluation["keys"][key]
        hs = _horizons(res)
        for j, (term, model, off) in enumerate(
            (("market", "market_recal", -0.1), ("llm", "llm", 0.1))
        ):
            xs, ms, los, his = [], [], [], []
            for i, h in enumerate(hs):
                e = res["by_horizon"][h]["forecasters"][key]["encompassing"]["full"]
                if "coef" not in e or e["se"].get(term) is None:
                    continue
                c, se = e["coef"][term], e["se"][term]
                xs.append(i + off)
                ms.append(c)
                los.append(c - z * se)
                his.append(c + z * se)
            color = COLORS[model]
            for x, lo, hi in zip(xs, los, his, strict=True):
                ax.plot([x, x], [lo, hi], color=color, lw=LW)
            label = "b (market)" if j == 0 else "c (LLM)"
            ax.plot(xs, ms, "o", color=color, ms=MS, mec=SURFACE, mew=1.2, label=label)
        ax.axhline(0, color=INK_2, lw=0.9)
        ax.set_xticks(range(len(hs)), hs)
        ax.set_xlim(-0.5, len(hs) - 0.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(short_key(key), loc="left", fontsize=9)
    _legend_below(fig, axes[0])
    fig.suptitle(
        "logit(y) = a + b·logit(market) + c·logit(LLM), full sample, 95% CI (event-clustered)",
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.08 / nrow, 1, 1))
    return _save(fig, out_dir, name)


def cumulative_pnl(
    evaluation: dict[str, Any],
    key: str,
    out_dir: Path,
    name: str = "pnl",
    models: Sequence[str] = ("market_recal", "pool", "llm"),
) -> list[Path]:
    """Cumulative P&L after fees over the test period (trades ordered by close time)."""
    import matplotlib.dates as mdates
    import numpy as np

    Figure = _mpl()
    pnl = evaluation["keys"][key].get("pnl", {})
    curves = pnl.get("curves", {})
    fig = Figure(figsize=(6.4, 3.3))
    ax = fig.add_subplot()
    for m in models:
        pts = curves.get(m) or []
        if not pts:
            continue
        last = {p["close_at"]: p["cum_pnl"] for p in pts}  # settle together: keep the last
        t = np.array([c.rstrip("Z") for c in last], dtype="datetime64[s]")
        cum = list(last.values())
        x = t.astype("datetime64[ms]").astype(object)
        ax.step(x, cum, where="post", color=COLORS[m], lw=LW, label=LABELS[m])
        ax.annotate(
            f"{cum[-1]:+,.0f}",
            (x[-1], cum[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=INK_2,
            fontsize=8,
        )
    ax.axhline(0, color=INK_2, lw=0.9)
    ax.set_ylabel("cumulative P&L after fees ($)")
    meta = pnl.get("meta", {})
    size = meta.get("sizing", {}).get("contracts", "")
    ax.set_title(
        f"Trading the test period: {short_key(key)}  ({size} contracts per signal, taker)",
        loc="left",
    )
    loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    _legend_below(fig, ax)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, out_dir, name)


def brier_by_horizon(
    evaluation: dict[str, Any],
    key: str,
    out_dir: Path,
    name: str = "brier_by_horizon",
    models: Sequence[str] = ("market", "market_recal", "llm", "pool"),
) -> list[Path]:
    """Test-period Brier score by horizon: the market sharpens as close nears."""
    Figure = _mpl()
    res = evaluation["keys"][key]
    hs = [h for h in _horizons(res) if _pool_block(res, key, h)]
    fig = Figure(figsize=(5.2, 3.3))
    ax = fig.add_subplot()
    for m in models:
        ys = [_pool_block(res, key, h).get(m, {}).get("brier") for h in hs]
        pts = [(i, y) for i, y in enumerate(ys) if y is not None]
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "-o",
            color=COLORS[m],
            lw=LW,
            ms=MS,
            mec=SURFACE,
            mew=1.2,
            label=LABELS[m],
        )
    ax.set_xticks(range(len(hs)), [f"{h} before close" for h in hs])
    ax.set_xlim(-0.3, len(hs) - 0.7)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title(f"Test-period Brier by horizon: {short_key(key)}", loc="left")
    _legend_below(fig, ax, ncol=2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return _save(fig, out_dir, name)


def all_figures(evaluation: dict[str, Any], out_dir: Path, featured: str) -> dict[str, list[Path]]:
    """Every report figure. ``featured`` is the forecaster|condition key shown in the
    single-forecaster figures (reliability, P&L, Brier by horizon)."""
    return {
        "forest": forest(evaluation, out_dir),
        "brier_by_horizon": brier_by_horizon(evaluation, featured, out_dir),
        "reliability": reliability(evaluation, featured, out_dir),
        "encompassing": encompassing(evaluation, out_dir),
        "pnl": cumulative_pnl(evaluation, featured, out_dir),
    }
