"""Funciones de visualización para diagramas de persistencia, barcodes y mapas."""
from __future__ import annotations

import folium
import numpy as np
import plotly.graph_objects as go
from folium.plugins import MarkerCluster
from pyproj import Transformer

from .data_loader import NIVEL_COLOR

UTM_TO_LATLON = Transformer.from_crs(32614, 4326, always_xy=True)


def utm_to_latlon(x: float, y: float) -> tuple[float, float]:
    lon, lat = UTM_TO_LATLON.transform(x, y)
    return lat, lon


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def persistence_diagram(dgms: list[np.ndarray], thresh: float) -> go.Figure:
    fig = go.Figure()
    lim = thresh * 1.05
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim], mode="lines", line=dict(color="gray", dash="dot"),
        name="diagonal", hoverinfo="skip", showlegend=False,
    ))
    colors = {0: "#1f77b4", 1: "#ff7f0e"}
    for dim, d in enumerate(dgms):
        if d is None or len(d) == 0:
            continue
        deaths = np.where(np.isfinite(d[:, 1]), d[:, 1], thresh)
        fig.add_trace(go.Scatter(
            x=d[:, 0], y=deaths, mode="markers",
            marker=dict(size=7, color=colors.get(dim, "gray"), opacity=0.7),
            name=f"H{dim}",
            hovertemplate=(f"H{dim}<br>birth: %{{x:.0f}} m"
                           f"<br>death: %{{y:.0f}} m<extra></extra>"),
        ))
    fig.update_layout(
        title="Diagrama de persistencia",
        xaxis_title="birth (m)", yaxis_title="death (m)",
        width=600, height=600,
        xaxis=dict(range=[-thresh*0.02, lim]),
        yaxis=dict(range=[-thresh*0.02, lim]),
    )
    return fig


def barcode(dgms: list[np.ndarray], thresh: float, dim: int = 1) -> go.Figure:
    fig = go.Figure()
    d = dgms[dim]
    if d is None or len(d) == 0:
        fig.update_layout(title=f"H{dim} — sin barras", height=200)
        return fig
    deaths = np.where(np.isfinite(d[:, 1]), d[:, 1], thresh)
    pers = deaths - d[:, 0]
    order = np.argsort(-pers)
    for i, idx in enumerate(order):
        fig.add_trace(go.Scatter(
            x=[d[idx, 0], deaths[idx]], y=[i, i], mode="lines",
            line=dict(color="#ff7f0e" if dim == 1 else "#1f77b4", width=2),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        title=f"Barcode H{dim} ({len(order)} barras)",
        xaxis_title="ε (m)", yaxis=dict(visible=False),
        height=max(200, min(20 * len(order), 600)),
    )
    return fig


def betti_curve_fig(eps: np.ndarray, betti: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=betti[:, 0], name="β₀ (componentes)",
                             line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=eps, y=betti[:, 1], name="β₁ (huecos)",
                             line=dict(color="#ff7f0e"), yaxis="y2"))
    fig.update_layout(
        title="Curvas de Betti",
        xaxis_title="ε (m)",
        yaxis=dict(title="β₀", color="#1f77b4"),
        yaxis2=dict(title="β₁", overlaying="y", side="right", color="#ff7f0e"),
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Mapas
# ---------------------------------------------------------------------------
def base_map(zoom: int = 11) -> folium.Map:
    return folium.Map(location=[19.4326, -99.1332], zoom_start=zoom, tiles="cartodbpositron")


def add_school_layer(m: folium.Map, df, sample_max: int = 1500) -> folium.Map:
    cluster = MarkerCluster().add_to(m)
    if len(df) > sample_max:
        df = df.sample(sample_max, random_state=0)
    for _, r in df.iterrows():
        folium.CircleMarker(
            [r["latitud"], r["longitud"]], radius=3,
            color=NIVEL_COLOR.get(r["nivel"], "gray"), fill=True, fill_opacity=0.7,
            tooltip=f"{r['nivel']} ({r['sector']}) — {str(r['nom_estab'])[:60]}",
        ).add_to(cluster)
    return m


def add_holes_layer(m: folium.Map, holes: list[dict], color: str = "red",
                    label: str = "huecos") -> folium.Map:
    layer = folium.FeatureGroup(name=label)
    for h in holes:
        lat, lon = utm_to_latlon(*h["centroid_xy"])
        # Círculo centrado en el centroide geométrico del hueco
        folium.Circle(
            [lat, lon], radius=max(h.get("geom_radius", h["pers"] / 2), 100),
            color=color, fill=True, fill_opacity=0.15,
            popup=(f"<b>{label}</b><br>birth: {h['birth']:.0f} m"
                   f"<br>death: {h['death']:.0f} m"
                   f"<br>persistencia: {h['pers']:.0f} m"
                   f"<br>vértices: {h['n_verts']}"),
        ).add_to(layer)
        folium.CircleMarker([lat, lon], radius=4, color=color, fill=True).add_to(layer)

        # Marcador de ubicación sugerida para nueva escuela
        if "nueva_escuela_xy" in h:
            lat2, lon2 = utm_to_latlon(*h["nueva_escuela_xy"])
            folium.Marker(
                [lat2, lon2],
                icon=folium.DivIcon(
                    html='<div style="font-size:22px;line-height:1;'
                         'color:#FFD700;text-shadow:0 0 3px #333;'
                         'cursor:pointer">★</div>',
                    icon_size=(22, 22),
                    icon_anchor=(11, 11),
                ),
                tooltip="📍 Ubicación sugerida para nueva escuela",
                popup=(
                    "<b>Ubicación sugerida para nueva escuela</b><br>"
                    "Punto más alejado de cualquier escuela existente "
                    "dentro del área del hueco.<br>"
                    "<i>Prioridad alta para nuevos planteles.</i>"
                ),
            ).add_to(layer)
    layer.add_to(m)
    return m
