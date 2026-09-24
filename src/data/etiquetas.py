"""Taxonomia anatomica del dataset, usada EXACTAMENTE como viene (seccion 2.1).

Codificacion de las mascaras de PENGWIN Task 1 (CT):
    0        fondo
    1..10    sacro          (1  = fragmento principal, 2..10  = fragmentos de fractura)
    11..20   coxal izquierdo (11 = principal,          12..20 = fragmentos)
    21..30   coxal derecho   (21 = principal,          22..30 = fragmentos)
Un hueso sin fractura es un unico fragmento (solo 1, 11 o 21).
No se subdivide ilion/isquion/pubis: no tiene ground truth verificable.

Verificar en el EDA de la semana 8 que los casos reales respetan esto
(scripts/eda.py lo revisa y avisa si aparece una etiqueta fuera de 0..30).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

REGIONES = ["SA", "LI", "RI"]                     # indices 0, 1, 2
NOMBRES = {"SA": "Sacro", "LI": "Coxal izquierdo", "RI": "Coxal derecho"}
# clases semanticas del modelo: 0 fondo, 1 SA, 2 LI, 3 RI
N_CLASES_SEM = 4
# rol del fragmento: 0 fondo, 1 principal, 2 secundario (conminuto)
N_ROLES = 3

# colores por macro-hueso (codificacion por color exigida en 3.2); tonos por fragmento
COLORES_REGION = {"SA": (230, 159, 0), "LI": (86, 180, 233), "RI": (0, 158, 115)}


def region_de_etiqueta(lbl):
    """0 fondo, 1 SA, 2 LI, 3 RI (funciona con escalares y arrays)."""
    lbl = np.asarray(lbl)
    reg = np.zeros_like(lbl, dtype=np.uint8)
    m = lbl > 0
    reg[m] = ((lbl[m] - 1) // 10 + 1).astype(np.uint8)
    return reg


def indice_en_region(lbl):
    """1..10 dentro de su region (1 = principal)."""
    lbl = np.asarray(lbl)
    idx = np.zeros_like(lbl, dtype=np.uint8)
    m = lbl > 0
    idx[m] = ((lbl[m] - 1) % 10 + 1).astype(np.uint8)
    return idx


def rol_de_etiqueta(lbl):
    """0 fondo, 1 principal, 2 secundario."""
    idx = indice_en_region(lbl)
    rol = np.zeros_like(idx, dtype=np.uint8)
    rol[idx == 1] = 1
    rol[idx >= 2] = 2
    return rol


def etiqueta(region: int, indice: int) -> int:
    """region 1..3, indice 1..10 -> etiqueta PENGWIN."""
    return (region - 1) * 10 + indice


def nombre_fragmento(lbl: int) -> str:
    reg = int(region_de_etiqueta(lbl))
    idx = int(indice_en_region(lbl))
    return f"{REGIONES[reg - 1]}-{idx}"


def mapa_borde(lbl2d: np.ndarray, grosor: int = 1) -> np.ndarray:
    """Pixeles de un fragmento que tocan OTRO fragmento (superficie de contacto).
    Es la version ligera del boundary-core de MIC-DKFZ / contact-surface map:
    le da al watershed semillas limpias para separar fragmentos pegados."""
    L = lbl2d.astype(np.int32)
    tam = 2 * grosor + 1
    grande = np.where(L > 0, L, 10_000)
    vecino_max = ndi.maximum_filter(L, size=tam)
    vecino_min_nz = ndi.minimum_filter(grande, size=tam)
    borde = (L > 0) & ((vecino_max > L) | (vecino_min_nz < L))
    return borde.astype(np.uint8)


def validar_etiquetas(vol: np.ndarray) -> dict:
    vals = np.unique(vol)
    fuera = [int(v) for v in vals if v < 0 or v > 30]
    return {"valores": [int(v) for v in vals], "fuera_de_rango": fuera}
