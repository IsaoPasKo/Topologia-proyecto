"""Panorama: filtros, conteos y mapa interactivo de escuelas."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from utils.data_loader import NIVEL_COLOR, NIVELES, SECTORES, load_escuelas
from utils.plotting import add_school_layer, base_map

st.set_page_config(page_title="Panorama", page_icon="📊", layout="wide")
st.title("📊 Panorama")

df = load_escuelas()

with st.sidebar:
    st.header("Filtros")
    nivs = st.multiselect("Nivel", NIVELES, default=NIVELES)
    secs = st.multiselect("Sector", SECTORES, default=SECTORES)
    alcs = st.multiselect("Alcaldía", sorted(df["municipio"].unique()),
                          default=sorted(df["municipio"].unique()))
    sample_n = st.slider("Máx. puntos a dibujar", 200, 5000, 1500, 100)

filt = df[
    df["nivel"].isin(nivs)
    & df["sector"].isin(secs)
    & df["municipio"].isin(alcs)
]
st.markdown(f"**{len(filt):,}** escuelas seleccionadas")

c1, c2 = st.columns([3, 2])

with c1:
    st.subheader("Mapa")
    if len(filt) == 0:
        st.info("No hay escuelas con los filtros actuales.")
    else:
        m = base_map()
        add_school_layer(m, filt, sample_max=sample_n)
        st_folium(m, height=520, use_container_width=True, returned_objects=[])

with c2:
    st.subheader("Distribución por nivel")
    if len(filt):
        nv_counts = filt["nivel"].value_counts().reset_index()
        nv_counts.columns = ["nivel", "n"]
        fig = px.bar(nv_counts, x="nivel", y="n", color="nivel",
                     color_discrete_map=NIVEL_COLOR)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top alcaldías")
        alc_counts = (
            filt["municipio"].value_counts().head(10).reset_index()
        )
        alc_counts.columns = ["alcaldía", "n"]
        fig2 = px.bar(alc_counts, x="n", y="alcaldía", orientation="h")
        fig2.update_layout(height=350, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)
