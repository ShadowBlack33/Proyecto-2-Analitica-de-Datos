"""Metricas de clasificacion y segmentacion.

Clasificacion (categoria anatomica): F1 macro y AUC macro sobre la presencia de SA/LI/RI
por corte, mas una matriz de confusion 4x4 (fondo/SA/LI/RI) a nivel de pixel para el informe.

Segmentacion de fragmento (en 3D, resolucion nativa): cada fragmento del GT se empareja
con un fragmento predicho de la MISMA region maximizando IoU (asignacion hungara).
Un fragmento GT sin pareja cuenta con Dice = IoU = 0 (no se esconden los fallos).
Extension opcional (puntos extra): HD95 y ASSD en mm por fragmento, con el spacing real.
"""
from __future__ import annotations

import numpy as np


def _media_nan(v):
    """nanmean sin avisos: NaN si no hay valores (p. ej. ningun fragmento emparejado)."""
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from ..data.etiquetas import nombre_fragmento, region_de_etiqueta


# ============================================================ clasificacion
# Implementadas con NumPy, sin scikit-learn: en Windows con Smart App Control /
# Application Control, scikit-learn falla al importar porque importa scipy.stats,
# y la politica bloquea uno de sus DLL. Estas funciones replican exactamente las
# de sklearn.metrics (verificadas contra sklearn en tests/test_basicos.py).

def f1_por_clase(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """F1 de cada columna de un problema multi-etiqueta (N, C). 0 si no hay positivos
    ni predichos (equivale a zero_division=0 de sklearn)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    if y_true.ndim == 1:
        y_true, y_pred = y_true[:, None], y_pred[:, None]
    tp = (y_true & y_pred).sum(0).astype(float)
    fp = (~y_true & y_pred).sum(0).astype(float)
    fn = (y_true & ~y_pred).sum(0).astype(float)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(f1_por_clase(y_true, y_pred)))


def _rangos_promedio(x: np.ndarray) -> np.ndarray:
    """Rangos 1..n con empates promediados (como scipy.stats.rankdata)."""
    orden = np.argsort(x, kind="mergesort")
    xs = x[orden]
    rangos = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        rangos[orden[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return rangos


def auc_binaria(y: np.ndarray, score: np.ndarray) -> float:
    """AUC-ROC por la formula de Mann-Whitney: probabilidad de que un positivo al azar
    tenga mayor score que un negativo al azar (empates cuentan 1/2)."""
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rangos_promedio(score)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_clases: int) -> np.ndarray:
    """Filas = verdad, columnas = prediccion (misma convencion de sklearn)."""
    idx = np.asarray(y_true, dtype=np.int64) * n_clases + np.asarray(y_pred, dtype=np.int64)
    return np.bincount(idx.ravel(), minlength=n_clases * n_clases).reshape(n_clases, n_clases)


def metricas_clasificacion(y_true: np.ndarray, y_prob: np.ndarray, umbral=0.5) -> dict:
    """y_true, y_prob (N, 3) presencia de SA/LI/RI por corte."""
    y_pred = (y_prob >= umbral).astype(int)
    aucs = [auc_binaria(y_true[:, c], y_prob[:, c]) for c in range(y_true.shape[1])
            if len(np.unique(y_true[:, c])) == 2]
    f1s = f1_por_clase(y_true, y_pred)
    return {
        "f1_macro": float(np.mean(f1s)),
        "f1_por_clase": dict(zip(["SA", "LI", "RI"], f1s.tolist())),
        "auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
    }


def matriz_confusion_pixel(sem_gt: np.ndarray, sem_pred: np.ndarray, submuestreo=4) -> np.ndarray:
    return matriz_confusion(sem_gt[..., ::submuestreo, ::submuestreo].ravel(),
                            sem_pred[..., ::submuestreo, ::submuestreo].ravel(), 4)


# ============================================================ segmentacion
def dice_iou(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(a, b).sum()
    sa, sb = a.sum(), b.sum()
    if sa + sb == 0:
        return 1.0, 1.0
    return float(2 * inter / (sa + sb)), float(inter / (sa + sb - inter))


def _superficie(m: np.ndarray) -> np.ndarray:
    return m & ~ndi.binary_erosion(m, structure=ndi.generate_binary_structure(3, 1))


def distancias_superficie(a: np.ndarray, b: np.ndarray, spacing) -> tuple[float, float]:
    """HD95 y ASSD en mm entre dos mascaras binarias 3D (misma definicion que medpy)."""
    if not a.any() or not b.any():
        return float("nan"), float("nan")
    idx = np.where(a | b)
    sl = tuple(slice(max(0, i.min() - 2), i.max() + 3) for i in idx)
    a, b = a[sl], b[sl]
    sa, sb = _superficie(a), _superficie(b)
    dt_b = ndi.distance_transform_edt(~sb, sampling=spacing)
    dt_a = ndi.distance_transform_edt(~sa, sampling=spacing)
    d_ab, d_ba = dt_b[sa], dt_a[sb]
    hd95 = max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))
    assd = (d_ab.sum() + d_ba.sum()) / (len(d_ab) + len(d_ba))
    return float(hd95), float(assd)


def metricas_fragmentos(etq_gt: np.ndarray, etq_pred: np.ndarray, spacing, superficie=True) -> dict:
    filas = []
    for r in (1, 2, 3):
        g_ids = [int(v) for v in np.unique(etq_gt) if v > 0 and region_de_etiqueta(v) == r]
        p_ids = [int(v) for v in np.unique(etq_pred) if v > 0 and region_de_etiqueta(v) == r]
        if not g_ids:
            continue
        iou = np.zeros((len(g_ids), max(1, len(p_ids))))
        for i, g in enumerate(g_ids):
            mg = etq_gt == g
            for j, p in enumerate(p_ids):
                iou[i, j] = dice_iou(mg, etq_pred == p)[1]
        fil, col = linear_sum_assignment(-iou)
        pareja = {g_ids[i]: p_ids[j] for i, j in zip(fil, col) if p_ids and iou[i, j] > 0}
        for g in g_ids:
            fila = {"gt": nombre_fragmento(g), "pred": None, "dice": 0.0, "iou": 0.0,
                    "hd95_mm": float("nan"), "assd_mm": float("nan")}
            if g in pareja:
                mg, mp = etq_gt == g, etq_pred == pareja[g]
                fila["pred"] = nombre_fragmento(pareja[g])
                fila["dice"], fila["iou"] = dice_iou(mg, mp)
                if superficie:
                    fila["hd95_mm"], fila["assd_mm"] = distancias_superficie(mg, mp, spacing)
            filas.append(fila)
    return {
        "dice_fragmento": float(np.mean([f["dice"] for f in filas])) if filas else float("nan"),
        "iou_fragmento": float(np.mean([f["iou"] for f in filas])) if filas else float("nan"),
        "hd95_mm": _media_nan([f["hd95_mm"] for f in filas]),
        "assd_mm": _media_nan([f["assd_mm"] for f in filas]),
        "detalle": filas,
    }
