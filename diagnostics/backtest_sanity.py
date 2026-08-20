"""Backtest honesty guards: placebo, look-ahead, and survivorship checks.

Run via ``python -m diagnostics.backtest_sanity`` (uses the synthetic store,
generating one if absent).  These are the tests that keep the backtest honest:

* PLACEBO — shuffle the composite scores within each cross-section and re-run
  the IC.  A real signal must collapse to ~0; a shuffled score that still
  "predicts" the label is a tell-tale of label leakage (e.g. the forward window
  bleeding into the feature snapshot).
* LOOK-AHEAD — confirm ``historical_loader.assert_no_lookahead`` raises when a
  feature row is dated after its as-of snapshot.
* SURVIVORSHIP — confirm the engine flags ``survivorship_safe=False`` when the
  point-in-time universe file is withheld.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
import historical_loader as hl
import backtest as bt

logger = logging.getLogger(__name__)


def placebo_test(result: bt.BacktestResult, seed: int = 0) -> dict:
    """Shuffle composite_score within each date, recompute IC. Expect ~0."""
    rng = np.random.default_rng(seed)
    shuffled = result.panel.copy()
    parts = []
    for _, sub in shuffled.groupby("date"):
        sub = sub.copy()
        sub["composite_score"] = rng.permutation(sub["composite_score"].to_numpy())
        parts.append(sub)
    shuffled = pd.concat(parts, ignore_index=True)
    placebo_ic = bt.information_coefficient(shuffled)
    real_ic = bt.information_coefficient(result.panel)
    real_dd = real_ic["ic_maxdd_mean"] or 0.0
    plc_dd = placebo_ic["ic_maxdd_mean"] or 0.0
    passed = abs(plc_dd) < 0.05 and abs(real_dd) > abs(plc_dd) + 0.03
    return {
        "real_ic_maxdd": real_dd,
        "placebo_ic_maxdd": plc_dd,
        "real_ic_return": real_ic["ic_return_mean"],
        "placebo_ic_return": placebo_ic["ic_return_mean"],
        "passed": bool(passed),
        "interpretation": (
            "PASS: shuffling collapses the IC, real signal survives."
            if passed else
            "FAIL: shuffled scores still predict — investigate label leakage."
        ),
    }


def lookahead_test(store: hl.HistoricalStore) -> dict:
    """Confirm the look-ahead assertion fires on a deliberately-future row."""
    dates = store.feature_dates
    if len(dates) < 2:
        return {"passed": None, "note": "not enough snapshot dates to test"}
    asof = dates[0]
    future = store.features[store.features["date"] == dates[1]].head(3)
    raised = False
    try:
        hl.assert_no_lookahead(future, asof)
    except hl.LookAheadError:
        raised = True
    # and the legitimate slice must NOT raise
    legit_ok = True
    try:
        hl.assert_no_lookahead(
            store.features[store.features["date"] <= asof], asof
        )
    except hl.LookAheadError:
        legit_ok = False
    return {
        "passed": bool(raised and legit_ok),
        "raised_on_future_row": raised,
        "passed_on_legit_slice": legit_ok,
    }


def survivorship_test(store: hl.HistoricalStore) -> dict:
    """A store WITHOUT a universe must produce survivorship_safe=False."""
    no_uni = hl.HistoricalStore(
        features=store.features, prices=store.prices, universe=None,
        ima_holdings=store.ima_holdings, delistings=store.delistings,
    )
    res = bt.run_backtest(
        no_uni, bt.BacktestConfig(run_clusters=False),
    )
    with_uni = store.has_universe
    return {
        "passed": bool((not res.survivorship_safe) and with_uni),
        "flagged_unsafe_without_universe": not res.survivorship_safe,
        "store_has_universe": with_uni,
    }


def run_all(seed: int = 0) -> dict:
    """Load (or synthesize) the store, run the full backtest, run every guard."""
    if not config.HIST_FEATURES_PARQUET.exists():
        logger.warning("No historical store found — generating a synthetic one.")
        store = hl.generate_synthetic_store()
    else:
        store = hl.load_historical_store()

    result = bt.run_backtest(store, bt.BacktestConfig(run_clusters=False))
    report = {
        "placebo": placebo_test(result, seed=seed),
        "lookahead": lookahead_test(store),
        "survivorship": survivorship_test(store),
    }
    report["all_passed"] = all(
        v.get("passed") for v in report.values() if isinstance(v, dict)
    )
    return report


def _print(report: dict) -> None:
    print("\n=== Backtest sanity report ===")
    plc = report["placebo"]
    print(f"\n[PLACEBO]  real IC(maxDD)={plc['real_ic_maxdd']:.4f}  "
          f"placebo={plc['placebo_ic_maxdd']:.4f}  -> "
          f"{'PASS' if plc['passed'] else 'FAIL'}")
    print(f"           {plc['interpretation']}")
    la = report["lookahead"]
    print(f"\n[LOOK-AHEAD]  raised on future row={la['raised_on_future_row']}  "
          f"legit slice ok={la['passed_on_legit_slice']}  -> "
          f"{'PASS' if la['passed'] else 'FAIL'}")
    sv = report["survivorship"]
    print(f"\n[SURVIVORSHIP]  flagged unsafe w/o universe="
          f"{sv['flagged_unsafe_without_universe']}  -> "
          f"{'PASS' if sv['passed'] else 'FAIL'}")
    print(f"\n>>> ALL PASSED: {report['all_passed']}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _print(run_all())
