"""Wrapper sobre ripser con caché para recálculo libre desde Streamlit."""
from __future__ import annotations

import hashlib

import numpy as np
import streamlit as st
from ripser import ripser
from sklearn.cluster import KMeans


def _hash_array(X: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest()


def landmark_sample(X: np.ndarray, max_n: int = 1500, seed: int = 0) -> np.ndarray:
    if len(X) <= max_n:
        return X
    km = KMeans(n_clusters=max_n, n_init=3, random_state=seed).fit(X)
    return km.cluster_centers_


@st.cache_resource(show_spinner=True, max_entries=8)
def compute_vr(_X_key: str, X: np.ndarray, thresh: float, max_n: int) -> dict:
    """Vietoris-Rips + persistencia. La clave de caché es _X_key + thresh + max_n."""
    Xs = landmark_sample(X, max_n=max_n)
    res = ripser(Xs, maxdim=1, thresh=thresh, do_cocycles=True)
    return {
        "X": Xs,
        "X_original_n": len(X),
        "dgms": res["dgms"],
        "cocycles": res["cocycles"],
        "thresh": thresh,
    }


def compute_for_points(X: np.ndarray, thresh: float = 3000.0, max_n: int = 1500) -> dict:
    key = f"{_hash_array(X)}_{thresh}_{max_n}"
    return compute_vr(key, X, thresh, max_n)


def betti_curves(dgms: list[np.ndarray], eps_grid: np.ndarray) -> np.ndarray:
    out = np.zeros((len(eps_grid), 2), dtype=int)
    for i, eps in enumerate(eps_grid):
        for dim in (0, 1):
            d = dgms[dim]
            if len(d) == 0:
                continue
            alive = (d[:, 0] <= eps) & (d[:, 1] > eps)
            out[i, dim] = int(alive.sum())
    return out


def _max_empty_point(X_all: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Punto interior del hueco más alejado de cualquier escuela.

    Construye una grilla densa dentro del casco convexo de los vértices del cociclo,
    filtra los candidatos que caen fuera del polígono y devuelve el que maximiza
    la distancia mínima a todos los puntos en X_all.
    """
    from scipy.spatial import cKDTree, ConvexHull
    from shapely.geometry import MultiPoint, Point

    pts = X_all[verts]
    lo, hi = pts.min(axis=0), pts.max(axis=0)

    # Grilla de ~60×60 dentro del bbox
    xs = np.linspace(lo[0], hi[0], 60)
    ys = np.linspace(lo[1], hi[1], 60)
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T  # (3600, 2)

    # Casco convexo de los vértices del cociclo como polígono de filtro
    try:
        hull_poly = MultiPoint(pts).convex_hull
        mask = np.array([hull_poly.contains(Point(p)) for p in grid])
        candidates = grid[mask]
    except Exception:
        candidates = grid

    # Si el polígono es demasiado pequeño y no contiene ningún punto de la grilla,
    # caer de vuelta a la grilla completa
    if len(candidates) == 0:
        candidates = grid

    tree = cKDTree(X_all)
    dists, _ = tree.query(candidates, k=1)
    return candidates[np.argmax(dists)]


def top_h1_with_cocycles(r: dict, k: int = 5) -> list[dict]:
    """Devuelve los k features H1 más persistentes con centroides en el interior del hueco."""
    dgm1 = r["dgms"][1]
    cocycles = r["cocycles"][1] if len(r.get("cocycles", [])) > 1 else []
    X = r["X"]
    if len(dgm1) == 0 or len(cocycles) == 0:
        return []
    deaths = np.where(np.isfinite(dgm1[:, 1]), dgm1[:, 1], r["thresh"])
    pers = deaths - dgm1[:, 0]
    order = np.argsort(-pers)[:k]
    out = []
    for idx in order:
        coc = cocycles[idx]
        verts = np.unique(coc[:, :2].ravel()).astype(int) if len(coc) else np.array([], dtype=int)
        if len(verts) == 0:
            continue
        # Centroide = punto interior más alejado de cualquier escuela (opción A)
        centroid = _max_empty_point(X, verts)
        # Radio = distancia desde el centroide a la escuela más cercana
        from scipy.spatial import cKDTree
        tree = cKDTree(X)
        geom_radius = float(tree.query(centroid, k=1)[0])
        out.append({
            "birth": float(dgm1[idx, 0]),
            "death": float(deaths[idx]),
            "pers": float(pers[idx]),
            "centroid_xy": centroid,
            "vert_xy": X[verts],
            "n_verts": int(len(verts)),
            "geom_radius": geom_radius,
        })
    return out
