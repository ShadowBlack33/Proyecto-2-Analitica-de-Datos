"""Metricas objetivas de calidad de imagen (EDA semana 8).

El profesor advirtio que el dataset ya trae imagenes de baja calidad (7 centros,
escaneres distintos). En vez de separar "feas" a ojo, se mide:
  - nitidez: varianza del Laplaciano en la ventana de hueso (baja = borroso)
  - ruido: desviacion estandar en el aire fuera del paciente (alta = ruidoso)
  - espesor de corte: spacing en z (cortes gruesos = menos detalle 3D)
Con estas medidas se parte el test en terciles de calidad y se reportan las metricas
por tercil: evidencia directa de cuanto cae el modelo con imagenes malas.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def varianza_laplaciano(corte01: np.ndarray, mascara: np.ndarray | None = None) -> float:
    lap = ndi.laplace(corte01.astype(np.float32))
    if mascara is not None and mascara.any():
        return float(lap[mascara].var())
    return float(lap.var())


def ruido_aire(hu_corte: np.ndarray, umbral_aire=-900) -> float:
    """Desviacion estandar (HU) de los pixeles de aire, erosionados para no tomar bordes."""
    aire = ndi.binary_erosion(hu_corte < umbral_aire, iterations=3)
    if aire.sum() < 500:
        return float("nan")
    return float(hu_corte[aire].std())


def perfil_calidad(hu_vol: np.ndarray, spacing_zyx, ventana=(-200, 1200), n_cortes=9) -> dict:
    """Resume la calidad de un volumen tomando n cortes repartidos en la zona con hueso."""
    lo, hi = ventana
    hueso = hu_vol > 200
    zs = np.where(hueso.reshape(hu_vol.shape[0], -1).any(1))[0]
    if len(zs) == 0:
        zs = np.arange(hu_vol.shape[0])
    elegidos = np.linspace(zs.min(), zs.max(), num=min(n_cortes, len(zs))).astype(int)
    lap, ruido = [], []
    for z in elegidos:
        c = np.clip((hu_vol[z].astype(np.float32) - lo) / (hi - lo), 0, 1)
        m = ndi.binary_dilation(hueso[z], iterations=2)
        lap.append(varianza_laplaciano(c, m))
        ruido.append(ruido_aire(hu_vol[z]))
    return {
        "nitidez_lap": float(np.nanmedian(lap)),
        "ruido_aire_hu": float(np.nanmedian(ruido)) if np.isfinite(ruido).any() else float("nan"),
        "espesor_corte_mm": float(spacing_zyx[0]),
        "pixel_mm": float(spacing_zyx[1]),
    }


def terciles(valores: dict[str, float], mayor_es_mejor=True) -> dict[str, str]:
    """{id: valor} -> {id: 'alta'|'media'|'baja'} calidad."""
    ids = list(valores)
    v = np.array([valores[i] for i in ids], dtype=float)
    q1, q2 = np.nanquantile(v, [1 / 3, 2 / 3])
    out = {}
    for i, x in zip(ids, v):
        nivel = 0 if x <= q1 else (1 if x <= q2 else 2)
        if not mayor_es_mejor:
            nivel = 2 - nivel
        out[i] = ["baja", "media", "alta"][nivel]
    return out
