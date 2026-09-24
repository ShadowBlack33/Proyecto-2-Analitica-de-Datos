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
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from ..data.etiquetas import nombre_fragmento, region_de_etiqueta


# ============================================================ clasificacion
def metricas_clasificacion(y_true: np.ndarray, y_prob: np.ndarray, umbral=0.5) -> dict:
    """y_true, y_prob (N, 3) presencia de SA/LI/RI por corte."""
    y_pred = (y_prob >= umbral).astype(int)
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) == 2:
            aucs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_por_clase": dict(zip(["SA", "LI", "RI"], f1_score(y_true, y_pred, average=None, zero_division=0).tolist())),
        "auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
    }


def matriz_confusion_pixel(sem_gt: np.ndarray, sem_pred: np.ndarray, submuestreo=4) -> np.ndarray:
    return confusion_matrix(sem_gt[..., ::submuestreo, ::submuestreo].ravel(),
                            sem_pred[..., ::submuestreo, ::submuestreo].ravel(), labels=[0, 1, 2, 3])


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
