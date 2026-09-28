"""Los 3 visualizadores obligatorios (seccion 3.4), como figuras Plotly reutilizables.

  1. vis_mip_raw          MIP del volumen crudo, solo hueso por umbral en HU. Sin modelo.
  2. vis_corte            overlay 2D de cajas + mascaras coloreadas + DISTANCIA en el corte.
  3. vis_reconstruccion   marching cubes (skimage) + Plotly Mesh3d, fragmentos coloreados por
                          macro-hueso y la DISTANCIA junto a la etiqueta de cada fragmento.

La distancia se dibuja dentro de los visualizadores 2 y 3, no solo en una tabla (regla 3.4).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import ndimage as ndi
from skimage import measure

from ..data.etiquetas import COLORES_REGION, REGIONES, indice_en_region, nombre_fragmento, region_de_etiqueta


def color_fragmento(lbl: int, alpha: float | None = None) -> str:
    """Color del macro-hueso; los fragmentos secundarios en tonos mas claros del mismo color."""
    r = int(region_de_etiqueta(lbl))
    k = int(indice_en_region(lbl)) - 1
    base = np.array(COLORES_REGION[REGIONES[r - 1]], float)
    c = base + (255 - base) * min(0.65, 0.18 * k)
    c = tuple(int(v) for v in c)
    return f"rgba({c[0]},{c[1]},{c[2]},{alpha})" if alpha is not None else f"rgb({c[0]},{c[1]},{c[2]})"


def _etiqueta_texto(fila: dict) -> str:
    if fila.get("es_principal"):
        return f"{fila['nombre']} (principal)"
    d = fila.get("distancia_mm")
    return f"{fila['nombre']} · {d:.1f} mm" if d is not None else fila["nombre"]


# ============================================================ 1. MIP raw
def vis_mip_raw(hu: np.ndarray, spacing, umbral_hu=200, angulo=0, submuestreo=2) -> go.Figure:
    """MIP del hueso: umbral en HU sobre el volumen SIN procesar; el paso clasico previo
    al modelo. El angulo rota el volumen alrededor del eje craneo-caudal para verlo en 3D."""
    v = hu[:, ::submuestreo, ::submuestreo].astype(np.float32)
    v = np.where(v >= umbral_hu, v, umbral_hu)
    if angulo:
        v = ndi.rotate(v, angulo, axes=(1, 2), reshape=False, order=1, mode="constant", cval=umbral_hu)
    mip = v.max(axis=1)[::-1]                       # proyeccion anteroposterior, craneo arriba
    aspecto = spacing[0] / (spacing[2] * submuestreo)
    fig = go.Figure(go.Heatmap(z=mip, colorscale="Greys_r", showscale=False))
    fig.update_layout(title=f"MIP del hueso (HU ≥ {umbral_hu}), rotación {angulo}°",
                      yaxis=dict(scaleanchor="x", scaleratio=aspecto, visible=False),
                      xaxis=dict(visible=False), margin=dict(l=0, r=0, t=40, b=0), height=520)
    return fig


# ============================================================ 2. corte a corte
def vis_corte(hu_corte: np.ndarray, etq_corte: np.ndarray, dets: list[dict], tabla: list[dict],
              ventana=(-200, 1200), opacidad=0.45, titulo="") -> go.Figure:
    lo, hi = ventana
    base = np.clip((hu_corte.astype(np.float32) - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(base[..., None] * 255, 3, axis=2)
    for lbl in np.unique(etq_corte):
        if lbl == 0:
            continue
        m = etq_corte == lbl
        c = np.array([int(x) for x in color_fragmento(int(lbl))[4:-1].split(",")], float)
        rgb[m] = (1 - opacidad) * rgb[m] + opacidad * c
    fig = go.Figure(go.Image(z=rgb.astype(np.uint8)))
    for d in dets:
        x1, y1, x2, y2 = d["caja"]
        col = "rgb({},{},{})".format(*COLORES_REGION[REGIONES[d["clase"]]])
        fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color=col, width=2))
        fig.add_annotation(x=x1, y=y1, text=f"{REGIONES[d['clase']]} {d['score']:.2f}", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(color=col, size=11), bgcolor="rgba(0,0,0,0.6)")
    por_lbl = {f["etiqueta"]: f for f in tabla}
    for lbl in np.unique(etq_corte):
        if lbl == 0 or int(lbl) not in por_lbl:
            continue
        ys, xs = np.where(etq_corte == lbl)
        fig.add_annotation(x=float(xs.mean()), y=float(ys.mean()), text=_etiqueta_texto(por_lbl[int(lbl)]),
                           showarrow=True, arrowhead=0, ax=30, ay=-25, font=dict(color="white", size=11),
                           bgcolor="rgba(0,0,0,0.7)")
    fig.update_layout(title=titulo, margin=dict(l=0, r=0, t=40, b=0), height=560,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ============================================================ 3. reconstruccion 3D
def vis_reconstruccion(etq: np.ndarray, spacing, tabla: list[dict], paso=2) -> go.Figure:
    """Una malla por fragmento (marching cubes con el spacing real -> ejes en mm)."""
    fig = go.Figure()
    por_lbl = {f["etiqueta"]: f for f in tabla}
    for lbl in sorted(int(v) for v in np.unique(etq) if v > 0):
        m = etq == lbl
        if m.sum() < 50:
            continue
        idx = np.where(m)
        sl = tuple(slice(max(0, i.min() - 1), i.max() + 2) for i in idx)
        sub = np.pad(m[sl], 1).astype(np.float32)
        try:
            verts, caras, _, _ = measure.marching_cubes(sub, 0.5, spacing=spacing, step_size=paso)
        except (ValueError, RuntimeError):
            continue
        off = np.array([(s.start - 1) * sp for s, sp in zip(sl, spacing)])
        verts = verts + off
        z, y, x = verts.T
        fila = por_lbl.get(lbl, {"nombre": nombre_fragmento(lbl), "es_principal": indice_en_region(lbl) == 1})
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=caras[:, 0], j=caras[:, 1], k=caras[:, 2],
                                color=color_fragmento(lbl), opacity=1.0 if fila.get("es_principal") else 0.9,
                                name=_etiqueta_texto(fila), showlegend=True, flatshading=False,
                                hovertemplate=_etiqueta_texto(fila) + "<extra></extra>"))
        if not fila.get("es_principal"):   # etiqueta flotante con la distancia junto al fragmento
            c = verts.mean(0)
            fig.add_trace(go.Scatter3d(x=[c[2]], y=[c[1]], z=[c[0]], mode="text", text=[_etiqueta_texto(fila)],
                                       textfont=dict(size=13, color="white"), showlegend=False, hoverinfo="skip"))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
                                 bgcolor="rgb(15,18,24)"),
                      paper_bgcolor="rgb(15,18,24)", font=dict(color="white"),
                      margin=dict(l=0, r=0, t=30, b=0), height=640, legend=dict(itemsizing="constant"))
    return fig
