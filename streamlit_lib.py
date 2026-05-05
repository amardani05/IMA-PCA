"""Shared helpers for the Streamlit dashboard.

Loads the same JSON exports the React webapp consumes — single source of
truth for the analytics. Run ``python main.py`` first to populate
``webapp/public/data/``; the Streamlit app then reads from there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import config

DATA_DIR: Path = config.PROJECT_ROOT / "webapp" / "public" / "data"
MACRO_DIR: Path = DATA_DIR / "macro"
PITCH_DIR: Path = DATA_DIR / "pitches"
META_PATH: Path = config.PROJECT_ROOT / "webapp" / "public" / "meta.json"
CHARTS_DIR: Path = config.PROJECT_ROOT / "webapp" / "public" / "charts"

TIER_COLORS: dict[str, str] = {
    "Stable":     "#2c7a4b",
    "Mainstream": "#d4a017",
    "Elevated":   "#b3001b",
    # Legacy fallbacks
    "Low Risk":   "#2c7a4b",
    "Moderate":   "#7fb069",
    "High":       "#e57a44",
    "Critical":   "#b3001b",
}


# =============================================================================
# Cached loaders — Streamlit reruns the script top-to-bottom on every
# interaction, so cache anything that hits disk.
# =============================================================================
@st.cache_data(show_spinner=False)
def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


def load_meta() -> dict | None:
    if not META_PATH.exists():
        return None
    return _read_json(str(META_PATH))


def load_universe() -> pd.DataFrame:
    rows = _read_json(str(DATA_DIR / "universe.json"))
    df = pd.DataFrame(rows)
    return df


def load_portfolio_report() -> pd.DataFrame:
    rows = _read_json(str(DATA_DIR / "portfolio.json"))
    return pd.DataFrame(rows)


def load_clusters() -> pd.DataFrame:
    return pd.DataFrame(_read_json(str(DATA_DIR / "clusters.json")))


def load_pca_summary() -> pd.DataFrame:
    return pd.DataFrame(_read_json(str(DATA_DIR / "pca_summary.json")))


def load_opportunities() -> pd.DataFrame:
    rows = _read_json(str(DATA_DIR / "opportunities.json"))
    if isinstance(rows, list):
        return pd.DataFrame(rows)
    return pd.DataFrame()


def load_drift() -> pd.DataFrame:
    rows = _read_json(str(DATA_DIR / "drift_alerts.json"))
    if isinstance(rows, list):
        return pd.DataFrame(rows)
    return pd.DataFrame()


def load_trajectory() -> dict | None:
    path = DATA_DIR / "trajectory.json"
    if not path.exists():
        return None
    return _read_json(str(path))


# Macro
def load_macro_summary() -> dict | None:
    p = MACRO_DIR / "macro_summary.json"
    return _read_json(str(p)) if p.exists() else None


def load_factor_metadata() -> dict | None:
    p = MACRO_DIR / "factor_metadata.json"
    return _read_json(str(p)) if p.exists() else None


def load_macro_timeframes() -> dict | None:
    p = MACRO_DIR / "timeframes.json"
    return _read_json(str(p)) if p.exists() else None


def load_rolling_betas() -> dict | None:
    p = MACRO_DIR / "rolling_betas.json"
    return _read_json(str(p)) if p.exists() else None


def load_stock_betas() -> dict | None:
    p = MACRO_DIR / "stock_betas.json"
    return _read_json(str(p)) if p.exists() else None


# Pitches
def load_pitch_index() -> list[dict]:
    p = PITCH_DIR / "index.json"
    if not p.exists():
        return []
    payload = _read_json(str(p))
    return payload.get("pitches", [])


def load_pitch(ticker: str) -> dict | None:
    p = PITCH_DIR / f"{ticker.upper()}.json"
    if not p.exists():
        return None
    return _read_json(str(p))


# =============================================================================
# UI helpers
# =============================================================================
def tier_pill(tier: str | None) -> str:
    """Return a small inline-styled HTML span for a tier label."""
    if not tier:
        return ""
    color = TIER_COLORS.get(tier, "#666")
    text_color = "#1a202c" if color == "#d4a017" else "#fff"
    return (
        f'<span style="background:{color};color:{text_color};'
        f"padding:2px 8px;border-radius:10px;font-size:11px;"
        f'font-weight:600;letter-spacing:0.3px">{tier}</span>'
    )


def headline_metric(label: str, value: str, sub: str | None = None) -> None:
    """Compact headline-stat block."""
    st.markdown(
        f"""
        <div style="background:#fff;border:1px solid #e4e7eb;border-radius:8px;
                    padding:12px 14px">
          <div style="font-size:11px;color:#5a6370;text-transform:uppercase;
                      letter-spacing:0.6px;font-weight:600">{label}</div>
          <div style="font-size:22px;font-weight:600;margin-top:4px">{value}</div>
          {f'<div style="font-size:11px;color:#5a6370;margin-top:2px">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_pipeline_outputs() -> bool:
    """Stop the page with an actionable error if the JSON exports don't exist."""
    if not META_PATH.exists():
        st.error(
            "Pipeline outputs not found at "
            f"`{META_PATH.parent}`. Run `python main.py` from the repo root "
            "first; the Streamlit app reads the same JSON exports the React "
            "webapp uses."
        )
        return False
    return True


def apply_styles() -> None:
    """Inject app-wide CSS — call once near the top of every page.

    Streamlit's ``[theme]`` config doesn't expose font-size; the only way to
    bump it globally is a ``<style>`` injection. Setting ``html { font-size }``
    re-bases every Streamlit element that uses ``rem`` units, so the whole
    app scales proportionally without one-off per-element overrides.
    """
    st.markdown(
        """
        <style>
            /* 20px base — Streamlit's elements use rem, so headers, buttons,
               inputs, and the sidebar all scale up proportionally from the
               default 16px. */
            html { font-size: 20px; }

            /* Sidebar tweaks: Streamlit's default sidebar is narrow; bump it
               a little so the stock-selector tree fits comfortably. */
            [data-testid="stSidebar"] {
                min-width: 340px;
                max-width: 380px;
            }

            /* Sidebar text density — the 20px base would make the checkbox
               tree wall-of-text; ratchet sidebar checkboxes back down. */
            [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
            [data-testid="stSidebar"] [data-testid="stCheckbox"] label p {
                font-size: 14px;
                line-height: 1.3;
            }
            [data-testid="stSidebar"] details summary {
                font-size: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
