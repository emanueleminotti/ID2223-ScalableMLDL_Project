# app.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Fixed config
# -----------------------------
POLLENS = ["alder_pollen", "birch_pollen", "grass_pollen", "mugwort_pollen"]
FORECAST_PATH = "latest_forecasts.json"
HORIZON_DAYS = 7  # ALWAYS 7


# -----------------------------
# Minimal CSS for cards (Streamlit-compatible)
# -----------------------------
CSS = """
<style>
.block-container {max-width: 1150px; padding-top: 1.2rem; padding-bottom: 2rem;}
.card {
  border: 1px solid rgba(49, 51, 63, 0.2);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  background: rgba(255,255,255,0.02);
  margin-bottom: 14px;
}
.card h3 { margin: 0 0 8px 0; font-size: 16px; }
.muted { color: rgba(49, 51, 63, 0.65); font-size: 12px; }
.pill {
  display:inline-block; padding: 5px 10px; border-radius:999px;
  border:1px solid rgba(49, 51, 63, 0.18);
  background: rgba(49, 51, 63, 0.04);
  font-size: 12px;
}
.sep { height: 8px; }
</style>
"""


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ForecastBundle:
    generated_at: Optional[str]
    forecasts: Dict[str, List[Tuple[date, float]]]


@dataclass
class DayView:
    d: date
    value: float
    score: int
    label: str
    suggestions: List[str]


# -----------------------------
# Helpers
# -----------------------------
def pretty_pollen(p: str) -> str:
    return p.replace("_pollen", "").replace("_", " ").title()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_risk(value: float, pollen_type: str) -> Tuple[int, str]:
    scales = {
        "alder_pollen": 20.0,
        "birch_pollen": 800.0,
        "grass_pollen": 25.0,
        "mugwort_pollen": 30.0,
    }
    s = scales.get(pollen_type, 50.0)
    score = int(round(clamp((value / s) * 100.0, 0, 100)))

    if score < 30:
        return score, "Low"
    if score < 65:
        return score, "Medium"
    return score, "High"


def label_emoji(label: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(label, "🟡")


def suggestions_for(label: str, d: date) -> List[str]:
    dow = d.strftime("%A")
    if label == "Low":
        return [
            f"{dow}: good conditions for outdoors.",
            "If you're sensitive, keep meds available just in case.",
        ]
    if label == "Medium":
        return [
            f"{dow}: consider limiting long outdoor activities at peak hours.",
            "Ventilate later in the day; keep windows closed in the morning.",
            "Sunglasses can help reduce eye irritation.",
        ]
    return [
        f"{dow}: higher risk — plan outdoors carefully.",
        "Prefer indoor workouts; if going out, consider mask + sunglasses.",
        "Shower and change clothes after being outside.",
    ]


# -----------------------------
# Data loading
# -----------------------------
def load_latest_forecasts(path: str) -> Optional[ForecastBundle]:
    fp = Path(path)
    if not fp.exists():
        return None
    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
        generated_at = raw.get("generated_at")
        fc = raw.get("forecasts", {})

        forecasts: Dict[str, List[Tuple[date, float]]] = {}
        for pollen_type, items in fc.items():
            series: List[Tuple[date, float]] = []
            for it in items:
                d = datetime.strptime(it["date"], "%Y-%m-%d").date()
                v = float(it["value"])
                series.append((d, v))
            series.sort(key=lambda x: x[0])
            forecasts[pollen_type] = series

        return ForecastBundle(generated_at=generated_at, forecasts=forecasts)
    except Exception:
        return None


def generate_placeholders() -> ForecastBundle:
    today = date.today()
    forecasts: Dict[str, List[Tuple[date, float]]] = {}
    for p in POLLENS:
        series = []
        base = random.random()
        for i in range(1, HORIZON_DAYS + 1):
            d = today + timedelta(days=i)

            if base < 0.25:
                v = max(0.0, random.gauss(0.6, 0.4))
            elif base < 0.7:
                v = max(0.0, random.gauss(6.0, 2.3))
            else:
                v = max(0.0, random.gauss(14.0, 5.0))

            if p == "birch_pollen":
                v *= random.choice([10, 25, 50, 80])

            series.append((d, float(v)))
        forecasts[p] = series

    return ForecastBundle(generated_at=None, forecasts=forecasts)


def build_day_views(bundle: ForecastBundle, pollen_type: str) -> List[DayView]:
    series = bundle.forecasts.get(pollen_type, [])[:HORIZON_DAYS]
    if len(series) == 0:
    # use the same placeholder bundle if present, else create once
        if "placeholder_bundle" not in st.session_state:
            st.session_state["placeholder_bundle"] = generate_placeholders()
        series = st.session_state["placeholder_bundle"].forecasts[pollen_type]


    out: List[DayView] = []
    for d, v in series:
        score, label = compute_risk(v, pollen_type)
        out.append(DayView(d=d, value=v, score=score, label=label, suggestions=suggestions_for(label, d)))
    return out

def risk_thresholds(pollen_type: str) -> tuple[float, float]:
    """
    Thresholds on *level* for background bands (Low/Medium/High).
    Tweak these later once you see typical ranges for each pollen.
    Returns (low_to_medium, medium_to_high).
    """
    # crude defaults, but they create a nice UX now
    return {
        "alder_pollen": (5.0, 12.0),
        "grass_pollen": (5.0, 12.0),
        "mugwort_pollen": (5.0, 12.0),
        "birch_pollen": (200.0, 600.0),
    }.get(pollen_type, (5.0, 12.0))


def label_from_value(value: float, pollen_type: str) -> str:
    t1, t2 = risk_thresholds(pollen_type)
    if value < t1:
        return "Low"
    if value < t2:
        return "Medium"
    return "High"


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Pollen Forecast", page_icon="🌿", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

bundle = load_latest_forecasts(FORECAST_PATH)

# If no real data, keep placeholders stable across reruns
using_placeholders = False
if bundle is None:
    using_placeholders = True
    if "placeholder_bundle" not in st.session_state:
        st.session_state["placeholder_bundle"] = generate_placeholders()
    bundle = st.session_state["placeholder_bundle"]
else:
    # If real data arrives, drop placeholders (optional)
    if "placeholder_bundle" in st.session_state:
        del st.session_state["placeholder_bundle"]


# Header
left, right = st.columns([3, 1])
with left:
    st.title("🌿 Pollen Forecast & Allergy Risk")
    st.caption("7-day rolling forecast • Updated daily")
with right:
    st.markdown(f"<span class='pill'>Status: <b>{'Placeholders' if using_placeholders else 'Live data'}</b></span>", unsafe_allow_html=True)
    if bundle.generated_at:
        st.caption(f"Last update: {bundle.generated_at}")

st.divider()

tabs = st.tabs([pretty_pollen(p) for p in POLLENS])

for tab, pollen_type in zip(tabs, POLLENS):
    with tab:
        days = build_day_views(bundle, pollen_type)

        # --- Keep selection in session_state (per pollen)
        key = f"selected_day_{pollen_type}"
        if key not in st.session_state:
            st.session_state[key] = 0  # first day

        # --- Prepare plot data
        xs = list(range(len(days)))
        xlabels = [d.d.strftime("%a\n%d %b") for d in days]
        y = [d.value for d in days]

        # Determine y-limits (so bands look good)
        ymax = max(max(y) * 1.15, 1.0)
        t1, t2 = risk_thresholds(pollen_type)

        # --- Plot
        fig = plt.figure(figsize=(9.5, 3.6))
        ax = fig.add_subplot(111)

        # Background bands: Low / Medium / High
        ax.axhspan(0, min(t1, ymax), alpha=0.12)
        ax.axhspan(min(t1, ymax), min(t2, ymax), alpha=0.12)
        ax.axhspan(min(t2, ymax), ymax, alpha=0.12)

        ax.plot(xs, y, marker="o", linewidth=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel("Forecasted level")
        ax.set_ylim(0, ymax)
        ax.grid(True, axis="y", alpha=0.25)

        # Highlight selected day
        sel = st.session_state[key]
        ax.scatter([sel], [y[sel]], s=120, zorder=5)

        st.pyplot(fig, clear_figure=True)

        # --- “Weather-like” row of clickable days
        st.markdown("### Next 7 days")
        chip_cols = st.columns(7)
        for i, d in enumerate(days):
            # Show emoji based on thresholded label (so it matches background bands)
            band_label = label_from_value(d.value, pollen_type)
            emoji = label_emoji(band_label)
            short = d.d.strftime("%a %d")

            # Use a button as a day-chip
            clicked = chip_cols[i].button(f"{emoji} {short}", key=f"{pollen_type}_chip_{i}", use_container_width=True)
            if clicked:
                st.session_state[key] = i
                sel = i

        st.divider()

        # --- Detail panel for selected day
        d = days[st.session_state[key]]
        band_label = label_from_value(d.value, pollen_type)
        st.subheader(f"{d.d.strftime('%A %d %B')} — details")
        c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
        c1.metric("Forecasted level", f"{d.value:.2f}")
        c2.metric("Risk", f"{label_emoji(band_label)} {band_label}")
        c3.metric("Risk score", f"{d.score}/100")
        st.progress(d.score / 100.0)

        st.markdown("#### Suggestions")
        for s in d.suggestions:
            st.write(f"• {s}")

