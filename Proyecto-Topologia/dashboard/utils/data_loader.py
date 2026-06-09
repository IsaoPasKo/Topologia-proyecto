"""Carga del parquet y artefactos TDA pre-computados."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "processed" / "escuelas_cdmx.parquet"
TDA_DIR = ROOT / "data" / "processed" / "tda_results"

NIVELES = ["preescolar", "primaria", "secundaria", "media_superior", "media_tecnica"]
SECTORES = ["público", "privado"]
NIVEL_COLOR = {
    "preescolar": "#2ca02c",
    "primaria": "#1f77b4",
    "secundaria": "#ff7f0e",
    "media_superior": "#d62728",
    "media_tecnica": "#9467bd",
}


@st.cache_data(show_spinner=False)
def load_escuelas() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


@st.cache_data(show_spinner=False)
def load_tda(nivel: str) -> dict | None:
    path = TDA_DIR / f"{nivel}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def available_tda_levels() -> list[str]:
    return sorted(p.stem for p in TDA_DIR.glob("*.pkl"))
