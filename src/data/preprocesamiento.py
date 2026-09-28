"""Preprocesamiento: ventaneo HU, redimensionado reversible y cache por caso.

Por que una cache: cada CT pesa 40-430 MB en .mha. Leerlo completo en cada epoca
es inviable con 16 GB de RAM. Se preprocesa UNA vez cada volumen a cortes de 256x256
(uint8 para la imagen, uint8 para la etiqueta, float16 para el target de distancia)
y el Dataset solo lee los cortes que necesita.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from .etiquetas import indice_en_region, region_de_etiqueta
from .io import leer_volumen


# ------------------------------------------------------------------ intensidad
def ventanear(hu: np.ndarray, ventana=(-200, 1200)) -> np.ndarray:
    """Recorta a la ventana [min, max] en HU y lleva a [0, 1]."""
    lo, hi = ventana
    x = (hu.astype(np.float32) - lo) / float(hi - lo)
    return np.clip(x, 0.0, 1.0)


# ------------------------------------------------------------------ geometria
def info_padding(h: int, w: int) -> dict:
    """Padding simetrico para llevar el corte a cuadrado sin deformar la anatomia."""
    lado = max(h, w)
    pt, pl = (lado - h) // 2, (lado - w) // 2
    return {"lado": lado, "pad": (pt, lado - h - pt, pl, lado - w - pl), "h": h, "w": w}


def a_cuadrado(vol: np.ndarray, info: dict, valor=0) -> np.ndarray:
    pt, pb, pl, pr = info["pad"]
    return np.pad(vol, ((0, 0), (pt, pb), (pl, pr)), constant_values=valor)


def redimensionar(vol: np.ndarray, tam: int, modo: str) -> np.ndarray:
    """vol (Z, H, W) -> (Z, tam, tam). modo 'nearest' para etiquetas (una interpolacion
    bilineal inventaria ids de fragmento que no existen), 'bilinear' para imagen."""
    t = torch.from_numpy(np.ascontiguousarray(vol)).float().unsqueeze(1)
    if modo == "nearest":
        out = F.interpolate(t, size=(tam, tam), mode="nearest-exact")
    else:
        out = F.interpolate(t, size=(tam, tam), mode="bilinear", align_corners=False, antialias=True)
    return out.squeeze(1).numpy()


def deshacer_redimension(vol: np.ndarray, info: dict, modo: str) -> np.ndarray:
    """Inverso exacto de a_cuadrado + redimensionar: lleva predicciones de 256x256
    a la resolucion nativa del CT, que es donde se mide en mm."""
    lado = info["lado"]
    t = torch.from_numpy(np.ascontiguousarray(vol)).float().unsqueeze(1)
    if modo == "nearest":
        out = F.interpolate(t, size=(lado, lado), mode="nearest-exact")
    else:
        out = F.interpolate(t, size=(lado, lado), mode="bilinear", align_corners=False)
    out = out.squeeze(1).numpy()
    pt, pb, pl, pr = info["pad"]
    return out[:, pt: lado - pb, pl: lado - pr]


# ------------------------------------------------------------------ target de distancia
def mapa_distancia_gt(lbl: np.ndarray, spacing_zyx) -> np.ndarray:
    """Para cada voxel de un fragmento secundario: distancia (mm, 3D) al voxel mas
    cercano del fragmento PRINCIPAL de su misma region. 0 fuera de secundarios.

    El minimo de este mapa dentro de un fragmento es exactamente la distancia minima
    borde a borde que pide el punto 3.3 (distance_transform_edt con el spacing real).
    Es tambien el target denso de la cabeza de regresion de distancia.
    """
    out = np.zeros(lbl.shape, dtype=np.float32)
    region = region_de_etiqueta(lbl)
    idx = indice_en_region(lbl)
    for r in (1, 2, 3):
        m_reg = region == r
        if not m_reg.any():
            continue
        principal = m_reg & (idx == 1)
        secundario = m_reg & (idx >= 2)
        if not principal.any() or not secundario.any():
            continue
        # recorte a la caja de la region: el principal esta dentro, asi que la EDT es exacta
        zz, yy, xx = np.where(m_reg)
        sl = (slice(zz.min(), zz.max() + 1), slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1))
        edt = ndi.distance_transform_edt(~principal[sl], sampling=spacing_zyx).astype(np.float32)
        sub = out[sl]
        sub[secundario[sl]] = edt[secundario[sl]]
        out[sl] = sub
    return out


# ------------------------------------------------------------------ cache por caso
def preparar_caso(ruta_img, ruta_lbl, dir_cache, tam=256, ventana=(-200, 1200), margen_z=15) -> dict:
    """Lee un caso, lo preprocesa y lo guarda en <dir_cache>/<id>/ como .npy sin comprimir.

    .npy sin comprimir permite abrirlo con mmap: el Dataset lee solo los cortes que
    pide, sin cargar el volumen entero a RAM (16 GB no alcanzan para 150 casos en memoria).
    Con etiquetas, se recorta en z al rango con hueso +/- margen_z cortes: el resto del CT
    (abdomen, piernas) no aporta al entrenamiento y ocupa disco.
    """
    import json

    from .io import id_caso

    cid = id_caso(ruta_img)
    hu, spacing, meta = leer_volumen(ruta_img)
    Z = hu.shape[0]
    info = info_padding(hu.shape[1], hu.shape[2])

    lbl = dist = None
    z0, z1 = 0, Z
    if ruta_lbl is not None:
        lbl, _, _ = leer_volumen(ruta_lbl)
        if lbl.shape != hu.shape:
            raise ValueError(f"{cid}: imagen {hu.shape} y etiqueta {lbl.shape} no coinciden")
        lbl = lbl.astype(np.uint8)
        zs = np.where(lbl.reshape(Z, -1).max(1) > 0)[0]
        if len(zs):
            z0, z1 = max(0, zs.min() - margen_z), min(Z, zs.max() + 1 + margen_z)
        dist = mapa_distancia_gt(lbl, spacing)[z0:z1]   # EDT sobre el volumen completo, luego recorte
        lbl = lbl[z0:z1]

    img = ventanear(hu[z0:z1], ventana)
    del hu
    img = redimensionar(a_cuadrado(img, info, valor=0.0), tam, "bilinear")
    img_u8 = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)

    d = Path(dir_cache) / cid
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "img.npy", img_u8)
    res = {"id": cid, "spacing_zyx": list(spacing), "forma_zyx": list(meta["forma_zyx"]),
           "pad": list(info["pad"]), "lado": info["lado"], "z0": int(z0), "z1": int(z1), "tam": tam}
    if lbl is not None:
        lbl_r = redimensionar(a_cuadrado(lbl, info), tam, "nearest").astype(np.uint8)
        dist_r = redimensionar(a_cuadrado(dist, info), tam, "nearest").astype(np.float16)
        np.save(d / "lbl.npy", lbl_r)
        np.save(d / "dist.npy", dist_r)
        res["cortes_con_hueso"] = int((lbl_r.reshape(lbl_r.shape[0], -1).max(1) > 0).sum())
        res["etiquetas"] = [int(v) for v in np.unique(lbl) if v > 0]
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


def info_geometria(meta: dict) -> dict:
    """Info de padding para deshacer_redimension a partir del meta.json de la cache."""
    h, w = meta["forma_zyx"][1], meta["forma_zyx"][2]
    return {"lado": int(meta["lado"]), "pad": tuple(int(v) for v in meta["pad"]), "h": h, "w": w}
