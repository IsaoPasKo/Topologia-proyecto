"""Visualización del complejo Vietoris-Rips sobre el mapa real de CDMX."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
from scipy.spatial import cKDTree
from streamlit_folium import st_folium
import folium

from utils.data_loader import NIVEL_COLOR, NIVELES, SECTORES, load_escuelas
from utils.tda import landmark_sample
from utils.plotting import base_map, utm_to_latlon

st.set_page_config(page_title="Complejos Simpliciales", page_icon="🔵", layout="wide")
st.title("🔵 Complejos Simpliciales — animación de ε")

df = load_escuelas()

with st.sidebar:
    st.header("Selección")
    niv = st.selectbox("Nivel", ["todas", *NIVELES], index=2)
    sec = st.selectbox("Sector", ["ambos", *SECTORES])
    max_pts = st.slider("Puntos máx (submuestreo)", 50, 600, 250, 50,
                        help="Más puntos = más aristas y mapa más lento.")
    eps = st.slider("ε — radio (m)", 0, 4000, 800, 50)
    show_disks = st.checkbox("Mostrar discos de radio ε/2 (Čech)", True)

sub = df.copy()
if niv != "todas":
    sub = sub[sub["nivel"] == niv]
if sec != "ambos":
    sub = sub[sub["sector"] == sec]
st.caption(f"{len(sub):,} escuelas en la selección")

if len(sub) == 0:
    st.info("Sin datos para la selección actual.")
    st.stop()

X = sub[["x_utm", "y_utm"]].values
Xs = landmark_sample(X, max_n=max_pts)

# Aristas del Vietoris-Rips: pares (i, j) con dist(i, j) <= eps
edges = []
if eps > 0 and len(Xs) > 1:
    tree = cKDTree(Xs)
    pairs = tree.query_pairs(r=eps)
    edges = list(pairs)

# Triángulos (2-simplices) — sólo si pocos puntos para no congelar el browser
triangles = []
if len(Xs) <= 200 and eps > 0:
    edge_set = set(edges)
    n = len(Xs)
    for i, j in edges:
        for k in range(j + 1, n):
            if (i, k) in edge_set and (j, k) in edge_set:
                triangles.append((i, j, k))

color = NIVEL_COLOR.get(niv, "#444")

# Convertir todos los puntos UTM → lat/lon
latlon = np.array([utm_to_latlon(x, y) for x, y in Xs])  # shape (n, 2)

# ── Construir el mapa Folium ──────────────────────────────────────────────────
m = base_map()

# Capa de triángulos (2-símplex)
if triangles:
    tri_layer = folium.FeatureGroup(name="Triángulos (2-símplex)", show=True)
    for i, j, k in triangles:
        folium.Polygon(
            locations=[latlon[i], latlon[j], latlon[k]],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.20,
            weight=0,
        ).add_to(tri_layer)
    tri_layer.add_to(m)

# Capa de aristas (1-símplex)
if edges:
    edge_layer = folium.FeatureGroup(name="Aristas (1-símplex)", show=True)
    for i, j in edges:
        folium.PolyLine(
            locations=[latlon[i], latlon[j]],
            color=color,
            weight=1.5,
            opacity=0.6,
        ).add_to(edge_layer)
    edge_layer.add_to(m)

# Capa de discos Čech (círculos de radio ε/2 en metros)
if show_disks and eps > 0:
    disk_layer = folium.FeatureGroup(name="Discos Čech (radio ε/2)", show=True)
    for lat, lon in latlon:
        folium.Circle(
            location=[lat, lon],
            radius=eps / 2,          # en metros
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.08,
            weight=0,
        ).add_to(disk_layer)
    disk_layer.add_to(m)

# Capa de vértices (escuelas)
vert_layer = folium.FeatureGroup(name="Escuelas (vértices)", show=True)
for lat, lon in latlon:
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        weight=1,
    ).add_to(vert_layer)
vert_layer.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# ── Renderizar ────────────────────────────────────────────────────────────────
st.markdown(
    f"**Vietoris-Rips** · ε = {eps} m &nbsp;|&nbsp; "
    f"|V| = {len(Xs)} &nbsp;|&nbsp; "
    f"|E| = {len(edges)} &nbsp;|&nbsp; "
    f"|T| = {len(triangles)}"
)
st_folium(m, height=620, use_container_width=True, returned_objects=[])

with st.expander("¿Cómo se lee esto?"):
    st.markdown("""
- Cada punto es una escuela ubicada en su posición real dentro de CDMX.
- A radio ε aparecen **aristas** entre escuelas cuya distancia es ≤ ε.
- Cuando 3 escuelas forman un triángulo cerrado, aparece la **cara sombreada**
  (2-símplex).
- Los **discos de radio ε/2** (opcional) muestran la intuición del complejo
  de Čech: dos discos se intersectan ⇔ hay una arista (muy parecido al
  Vietoris-Rips en el plano).
- Mueve el slider para ver cómo nacen y se cierran los **huecos**.
- Usa el control de capas (esquina superior derecha del mapa) para
  mostrar u ocultar triángulos, aristas y discos por separado.
""")