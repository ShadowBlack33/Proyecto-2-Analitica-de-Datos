"""Medicion de la distancia de separacion (seccion 3.3).

Para cada fragmento secundario (conminuto): distancia minima borde a borde, en mm, al
fragmento PRINCIPAL del mismo hueso, con scipy.ndimage.distance_transform_edt y el
spacing real del header. Se aplica igual sobre la prediccion y sobre el ground truth,
para medir cuanto error de segmentacion se transfiere a la medida clinica.

Nota de resolucion (para el informe): la EDT mide entre centros de voxel, asi que dos
fragmentos en contacto dan ~1 voxel (0.5-1 mm), no 0. Es el piso de resolucion del metodo.

Ademas se reporta la distancia de la CUARTA CABEZA (regresion densa): minimo del mapa
predicho dentro de cada fragmento. Comparar ambas contra el GT es parte del analisis.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..data.etiquetas import indice_en_region, nombre_fragmento, region_de_etiqueta


def _caja(m, margen=1):
    idx = np.where(m)
    return tuple(slice(max(0, int(i.min()) - margen), int(i.max()) + 1 + margen) for i in idx)


def tabla_fragmentos(etq: np.ndarray, spacing, mapa_dist_pred: np.ndarray | None = None,
                     percentil: float = 0) -> list[dict]:
    """etq (Z,H,W) con etiquetas PENGWIN. mapa_dist_pred: log(1+mm) de la cabeza de distancia
    a resolucion nativa (opcional). Devuelve una fila por fragmento."""
    vox_ml = float(np.prod(spacing)) / 1000.0
    filas = []
    regiones = region_de_etiqueta(etq)
    for r in (1, 2, 3):
        m_reg = regiones == r
        if not m_reg.any():
            continue
        sl = _caja(m_reg)
        sub = etq[sl]
        principal = indice_en_region(sub) == 1
        edt = None
        if principal.any():
            edt = ndi.distance_transform_edt(~principal, sampling=spacing)
        for lbl in sorted(int(v) for v in np.unique(sub) if v > 0 and region_de_etiqueta(v) == r):
            m = sub == lbl
            zz, yy, xx = np.where(m)
            fila = {
                "etiqueta": lbl,
                "nombre": nombre_fragmento(lbl),
                "region": r,
                "es_principal": bool(indice_en_region(lbl) == 1),
                "voxeles": int(m.sum()),
                "volumen_ml": round(float(m.sum()) * vox_ml, 2),
                "centroide_zyx": [float(zz.mean() + sl[0].start), float(yy.mean() + sl[1].start),
                                  float(xx.mean() + sl[2].start)],
                "distancia_mm": None,
                "distancia_cabeza_mm": None,
            }
            if not fila["es_principal"] and edt is not None:
                vals = edt[m]
                fila["distancia_mm"] = round(float(np.percentile(vals, percentil) if percentil else vals.min()), 2)
                if mapa_dist_pred is not None:
                    pv = mapa_dist_pred[sl][m]
                    fila["distancia_cabeza_mm"] = round(float(np.expm1(np.percentile(pv, max(percentil, 1)))), 2)
            filas.append(fila)
    return filas


def comparar_con_gt(filas_pred: list[dict], filas_gt: list[dict], etq_pred, etq_gt) -> list[dict]:
    """Empareja cada fragmento secundario del GT con el fragmento predicho de mayor
    solapamiento (misma region) y reporta el error de distancia en mm."""
    res = []
    for g in filas_gt:
        if g["es_principal"]:
            continue
        m_gt = etq_gt == g["etiqueta"]
        cand = etq_pred[m_gt]
        cand = cand[(cand > 0) & (region_de_etiqueta(cand) == g["region"])]
        fila = {"gt": g["nombre"], "dist_gt_mm": g["distancia_mm"], "pred": None,
                "dist_pred_mm": None, "dist_cabeza_mm": None, "error_mm": None, "error_cabeza_mm": None}
        if len(cand):
            lbl = int(np.bincount(cand).argmax())
            p = next((f for f in filas_pred if f["etiqueta"] == lbl), None)
            if p is not None and not p["es_principal"]:
                fila.update(pred=p["nombre"], dist_pred_mm=p["distancia_mm"], dist_cabeza_mm=p["distancia_cabeza_mm"])
                if p["distancia_mm"] is not None and g["distancia_mm"] is not None:
                    fila["error_mm"] = round(abs(p["distancia_mm"] - g["distancia_mm"]), 2)
                if p["distancia_cabeza_mm"] is not None and g["distancia_mm"] is not None:
                    fila["error_cabeza_mm"] = round(abs(p["distancia_cabeza_mm"] - g["distancia_mm"]), 2)
        res.append(fila)
    return res
