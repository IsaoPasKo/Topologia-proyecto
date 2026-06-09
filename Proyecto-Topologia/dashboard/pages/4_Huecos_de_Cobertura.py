"""Huecos H1 persistentes proyectados sobre el mapa de CDMX."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from utils.data_loader import NIVEL_COLOR, SECTORES, load_escuelas, load_tda
from utils.plotting import add_holes_layer, base_map, utm_to_latlon
from utils.tda import compute_for_points, top_h1_with_cocycles

st.set_page_config(page_title="Huecos de Cobertura", page_icon="🗺️", layout="wide")
st.title("🗺️ Huecos de cobertura escolar")

st.markdown(
    "Los **huecos H₁ más persistentes** representan zonas rodeadas por "
    "escuelas pero con un déficit interno de cobertura. Cada hueco se dibuja "
    "como un círculo centrado en el centroide del ciclo, con radio = "
    "persistencia / 2."
)

df = load_escuelas()

with st.sidebar:
    st.header("Selección")
    niveles_disponibles = ["preescolar", "primaria", "secundaria",
                           "media_superior", "media_tecnica", "todas"]
    seleccion = st.multiselect("Niveles a mostrar", niveles_disponibles,
                               default=["primaria", "secundaria"])
    sector_sel = st.radio(
        "Sector",
        ["ambos", *SECTORES],
        index=0,
        horizontal=True,
        help="«ambos» usa los resultados pre-computados (más rápido). "
             "Filtrar por sector recalcula la persistencia sobre el subconjunto.",
    )
    if sector_sel != "ambos":
        thresh_recalc = st.slider("Umbral ε para recálculo (m)", 500, 10000, 10000, 100)
        st.caption("Cálculo sin submuestreo: se usan todas las escuelas del subconjunto.")
    top_k = st.slider("Top-K huecos por nivel", 1, 10, 5)
    show_schools = st.checkbox("Mostrar escuelas como puntos", False)

m = base_map()

rows = []
for niv in seleccion:
    if sector_sel == "ambos":
        r = load_tda(niv)
    else:
        sub = df if niv == "todas" else df[df["nivel"] == niv]
        sub = sub[sub["sector"] == sector_sel]
        if len(sub) < 3:
            st.warning(f"Pocas escuelas para `{niv}` / `{sector_sel}` "
                       f"({len(sub)}); se omite.")
            continue
        pts = sub[["x_utm", "y_utm"]].values
        with st.spinner(f"Calculando huecos para {niv} ({sector_sel}) "
                        f"con {len(pts):,} escuelas..."):
            r = compute_for_points(
                pts,
                thresh=thresh_recalc,
                max_n=len(pts),
            )
    if r is None:
        continue
    holes = top_h1_with_cocycles(r, k=top_k)
    color = NIVEL_COLOR.get(niv, "black")
    label_suffix = niv if sector_sel == "ambos" else f"{niv} · {sector_sel}"
    add_holes_layer(m, holes, color=color, label=f"huecos — {label_suffix}")
    for h in holes:
        lat, lon = utm_to_latlon(*h["centroid_xy"])
        rows.append({
            "nivel": niv,
            "sector": sector_sel,
            "lat": lat, "lon": lon,
            "birth (m)": round(h["birth"], 0),
            "death (m)": round(h["death"], 0),
            "persistencia (m)": round(h["pers"], 0),
            "n_vértices": h["n_verts"],
        })

if show_schools:
    import folium
    df_pts = df if sector_sel == "ambos" else df[df["sector"] == sector_sel]
    layer = folium.FeatureGroup(name=f"escuelas (muestra · {sector_sel})")
    for _, row in df_pts.sample(min(800, len(df_pts)), random_state=0).iterrows():
        folium.CircleMarker(
            [row["latitud"], row["longitud"]], radius=2,
            color=NIVEL_COLOR.get(row["nivel"], "gray"),
            fill=True, fill_opacity=0.6, weight=0,
        ).add_to(layer)
    layer.add_to(m)

import folium
folium.LayerControl(collapsed=False).add_to(m)

col1, col2 = st.columns([3, 2])

with col1:
    st_folium(m, height=620, use_container_width=True, returned_objects=[])

with col2:
    st.subheader("Huecos identificados")
    if rows:
        st.dataframe(
            pd.DataFrame(rows).sort_values("persistencia (m)", ascending=False),
            use_container_width=True, hide_index=True,
        )
        biggest = max(rows, key=lambda r: r["persistencia (m)"])
        st.info(
            f"**Hueco más significativo**: nivel `{biggest['nivel']}`, "
            f"~{int(biggest['persistencia (m)'])} m de persistencia, "
            f"centroide en ({biggest['lat']:.4f}, {biggest['lon']:.4f}). "
            "Indica una zona con déficit potencial de cobertura escolar "
            "que podría priorizarse para nuevos planteles o reasignación "
            "de matrícula."
        )
    else:
        st.info("Selecciona al menos un nivel para ver huecos.")
