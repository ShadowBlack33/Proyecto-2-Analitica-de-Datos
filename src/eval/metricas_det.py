"""Metricas de deteccion (seccion 5): IoU promedio, mAP@0.50 y mAP@[0.50:0.95].

Mismas definiciones que en S6/S7: TP/FP con umbral de IoU, emparejamiento por clase
(cada GT se empareja una sola vez, en orden de score), curva PR y AP por interpolacion
de 11 puntos. mAP@[0.50:0.95] = promedio sobre 10 umbrales de IoU (estilo COCO).
"""
from __future__ import annotations

import numpy as np


def iou_cajas(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ap_11_puntos(recall: np.ndarray, precision: np.ndarray) -> float:
    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p = precision[recall >= r].max() if (recall >= r).any() else 0.0
        ap += p / 11
    return float(ap)


def emparejar_por_clase(preds: list[list[dict]], gts: list[list[dict]], clase: int, umbral: float):
    """preds/gts: por imagen, listas de {'clase','caja',('score')}. Devuelve (scores, es_tp, n_gt)."""
    registros = []
    n_gt = 0
    for p_img, g_img in zip(preds, gts):
        g = [x for x in g_img if x["clase"] == clase]
        n_gt += len(g)
        usados = [False] * len(g)
        for d in sorted([x for x in p_img if x["clase"] == clase], key=lambda x: -x["score"]):
            mejor, j_mejor = 0.0, -1
            for j, gg in enumerate(g):
                if not usados[j]:
                    v = iou_cajas(d["caja"], gg["caja"])
                    if v > mejor:
                        mejor, j_mejor = v, j
            tp = mejor >= umbral and j_mejor >= 0
            if tp:
                usados[j_mejor] = True
            registros.append((d["score"], tp))
    registros.sort(key=lambda x: -x[0])
    return np.array([s for s, _ in registros]), np.array([t for _, t in registros], bool), n_gt


def ap_clase(preds, gts, clase, umbral=0.5) -> float:
    scores, tp, n_gt = emparejar_por_clase(preds, gts, clase, umbral)
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    return ap_11_puntos(ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9))


def resumen_deteccion(preds, gts, n_clases=3) -> dict:
    ap50 = [ap_clase(preds, gts, c, 0.5) for c in range(n_clases)]
    ap_coco = []
    for u in np.arange(0.5, 0.96, 0.05):
        ap_coco.append(np.nanmean([ap_clase(preds, gts, c, u) for c in range(n_clases)]))
    ious = []  # IoU promedio: mejor prediccion de la misma clase para cada GT
    for p_img, g_img in zip(preds, gts):
        for g in g_img:
            cand = [iou_cajas(p["caja"], g["caja"]) for p in p_img if p["clase"] == g["clase"]]
            ious.append(max(cand) if cand else 0.0)
    return {
        "iou_promedio": float(np.mean(ious)) if ious else float("nan"),
        "mAP@0.50": float(np.nanmean(ap50)),
        "mAP@[0.50:0.95]": float(np.nanmean(ap_coco)),
        "AP@0.50_por_clase": {n: v for n, v in zip(["SA", "LI", "RI"], ap50)},
    }
