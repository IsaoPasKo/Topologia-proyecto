"""Funciones de visualización para diagramas de persistencia, barcodes y mapas."""
from __future__ import annotations

import folium
import numpy as np
import plotly.graph_objects as go
from folium.plugins import MarkerCluster
from pyproj import Transformer
from scipy.spatial import cKDTree

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

        # Aristas del ciclo H₁ — borde topológico exacto del hueco (líneas punteadas)
        if "cycle_edges_xy" in h:
            for p1, p2 in h["cycle_edges_xy"]:
                lat1, lon1 = utm_to_latlon(*p1)
                lat2, lon2 = utm_to_latlon(*p2)
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color=color, weight=2, opacity=0.8, dash_array="6 4",
                    tooltip="Arista del ciclo H₁",
                ).add_to(layer)

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


def add_simplex_layer(
    m: folium.Map,
    Xs: np.ndarray,
    eps: float,
    color: str = "#1f77b4",
    label: str = "símplice",
    show_disks: bool = False,
) -> folium.Map:
    """Dibuja el complejo Vietoris-Rips a radio eps sobre el mapa.

    Xs: coordenadas UTM (n, 2) de los landmarks.
    Añade capas separadas para triángulos, aristas, discos y vértices.
    Los triángulos sólo se dibujan si len(Xs) <= 200 para no saturar el browser.
    """
    if len(Xs) < 2 or eps <= 0:
        return m

    latlon = np.array([utm_to_latlon(x, y) for x, y in Xs])

    tree = cKDTree(Xs)
    edges = list(tree.query_pairs(r=eps))

    # Solo dibujar triángulos si hay pocos puntos y pocas aristas para no congelar el browser
    triangles = []
    if len(Xs) <= 200 and len(edges) <= 4000:
        edge_set = set(edges)
        n = len(Xs)
        for i, j in edges:
            for k in range(j + 1, n):
                if (i, k) in edge_set and (j, k) in edge_set:
                    triangles.append((i, j, k))

    if triangles:
        tri_layer = folium.FeatureGroup(name=f"Triángulos — {label}", show=True)
        for i, j, k in triangles:
            folium.Polygon(
                locations=[latlon[i], latlon[j], latlon[k]],
                color=color, fill=True, fill_color=color,
                fill_opacity=0.15, weight=0,
            ).add_to(tri_layer)
        tri_layer.add_to(m)

    if edges:
        edge_layer = folium.FeatureGroup(name=f"Aristas — {label}", show=True)
        for i, j in edges:
            folium.PolyLine(
                locations=[latlon[i], latlon[j]],
                color=color, weight=1.5, opacity=0.55,
            ).add_to(edge_layer)
        edge_layer.add_to(m)

    if show_disks:
        disk_layer = folium.FeatureGroup(name=f"Discos Čech — {label}", show=False)
        for lat, lon in latlon:
            folium.Circle(
                location=[lat, lon], radius=eps / 2,
                color=color, fill=True, fill_color=color,
                fill_opacity=0.06, weight=0,
            ).add_to(disk_layer)
        disk_layer.add_to(m)

    vert_layer = folium.FeatureGroup(name=f"Vértices — {label}", show=True)
    for lat, lon in latlon:
        folium.CircleMarker(
            location=[lat, lon], radius=3,
            color=color, fill=True, fill_color=color,
            fill_opacity=0.85, weight=1,
        ).add_to(vert_layer)
    vert_layer.add_to(m)

    return m
