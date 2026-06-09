"""Entrada principal del dashboard.

Ejecuta:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permitir `from utils...` cuando streamlit corre desde la raíz del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from utils.data_loader import load_escuelas, NIVEL_COLOR

st.set_page_config(
    page_title="TDA de Escuelas — CDMX",
    page_icon="🏫",
    layout="wide",
)

st.title("🏫 Análisis topológico de escuelas en CDMX")
st.caption("Reto MA2007B — Geometría y Topología para Ciencia de Datos · Datos: DENUE 2025 (INEGI)")

df = load_escuelas()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Escuelas totales", f"{len(df):,}")
col2.metric("Niveles", df["nivel"].nunique())
col3.metric("Alcaldías cubiertas", df["municipio"].nunique())
col4.metric("Sectores", df["sector"].nunique())

st.markdown(
    """
**Objetivo.** Aplicar herramientas de análisis topológico (complejos
Vietoris-Rips y homología persistente) sobre la red de escuelas de la
Ciudad de México para identificar **patrones de conectividad y huecos
de cobertura** que el clustering clásico no revela.

### Navegación

- **📊 Panorama** — distribución espacial y conteos.
- **🔵 Complejos Simpliciales** — animación del complejo a medida que crece ε.
- **📈 Persistencia** — diagramas, barcode y curvas de Betti por nivel.
- **🗺️ Huecos de Cobertura** — features H1 más persistentes sobre el mapa.
"""
)

st.subheader("Conteos por nivel y sector")
ct = df.groupby(["nivel", "sector"]).size().unstack(fill_value=0)
ct["total"] = ct.sum(axis=1)
st.dataframe(ct.sort_values("total", ascending=False), use_container_width=True)

st.subheader("Categorías DENUE incluidas")
st.dataframe(
    df["nombre_act"].value_counts().rename_axis("Actividad SCIAN").reset_index(name="conteo"),
    use_container_width=True, hide_index=True,
)

with st.expander("Leyenda de colores por nivel"):
    for n, c in NIVEL_COLOR.items():
        st.markdown(f"<span style='color:{c}'>●</span> **{n}**", unsafe_allow_html=True)
