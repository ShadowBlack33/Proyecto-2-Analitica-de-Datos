"""Pipeline de inferencia de un CT completo: volumen -> fragmentos 3D con distancias en mm.

Corre en GPU o CPU (requisito 4.3). Pasos:
  1. ventaneo HU + padding a cuadrado + redimension a 256 (igual que en entrenamiento:
     leccion de la sesion de YOLO, un tamano distinto en inferencia pierde las cajas)
  2. forward corte a corte (2.5D) por lotes, con AMP en GPU
  3. NMS propio -> cajas por corte, llevadas a coordenadas nativas
  4. mapas densos -> resolucion nativa -> separacion de instancias 3D -> etiquetas PENGWIN
  5. tabla de fragmentos con distancia (geometrica y de la cabeza) en mm
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.dataset import pila_contexto
from .data.preprocesamiento import a_cuadrado, deshacer_redimension, info_padding, redimensionar, ventanear
from .models.modelo import PengwinNet
from .postprocess.distancia import tabla_fragmentos
from .postprocess.instancias import volumen_etiquetas
from .postprocess.nms import detecciones


def cargar_modelo(ruta_ckpt: str | Path, cfg: dict, device) -> PengwinNet:
    ck = torch.load(ruta_ckpt, map_location="cpu", weights_only=False)
    modelo = PengwinNet(cfg["modelo"], in_ch=2 * cfg["datos"]["contexto"] + 1)
    modelo.load_state_dict(ck["modelo"] if "modelo" in ck else ck)
    return modelo.to(device).eval()


def _a_nativo_por_bloques(vol256: np.ndarray, info: dict, modo: str, dtype, bloque=32) -> np.ndarray:
    partes = [deshacer_redimension(vol256[i:i + bloque], info, modo).astype(dtype)
              for i in range(0, vol256.shape[0], bloque)]
    return np.concatenate(partes, 0)


@torch.no_grad()
def inferir_volumen(modelo: PengwinNet, hu: np.ndarray, spacing, cfg: dict, device,
                    batch=8, amp=True, img01_256: np.ndarray | None = None) -> dict:
    t0 = time.perf_counter()
    tam, ctx = cfg["datos"]["resolucion"], cfg["datos"]["contexto"]
    Z, H, W = hu.shape
    info = info_padding(H, W)
    if img01_256 is None:
        img = ventanear(hu, cfg["datos"]["ventana_hu"])
        img01_256 = redimensionar(a_cuadrado(img, info, 0.0), tam, "bilinear").astype(np.float32)
    t_pre = time.perf_counter() - t0

    usar_amp = amp and device.type == "cuda"
    cls_p = np.zeros((Z, 6), np.float32)
    sem = np.zeros((Z, tam, tam), np.uint8)
    rol = np.zeros((Z, tam, tam), np.uint8)
    borde = np.zeros((Z, tam, tam), np.float16)
    dist = np.zeros((Z, tam, tam), np.float16)
    dets_256 = []
    dc = cfg["deteccion"]

    t1 = time.perf_counter()
    for z0 in range(0, Z, batch):
        zs = range(z0, min(Z, z0 + batch))
        x = torch.from_numpy(np.stack([pila_contexto(img01_256, z, ctx) for z in zs])).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp):
            o = modelo(x)
        cls_p[z0:z0 + len(zs)] = o["cls"].float().sigmoid().cpu().numpy()
        sem[z0:z0 + len(zs)] = o["sem"].argmax(1).cpu().numpy()
        rol[z0:z0 + len(zs)] = o["rol"].argmax(1).cpu().numpy()
        borde[z0:z0 + len(zs)] = o["borde"][:, 0].float().sigmoid().cpu().numpy()
        dist[z0:z0 + len(zs)] = o["dist"][:, 0].float().cpu().numpy()
        dets_256 += detecciones(o["det_cls"], o["det_box"], dc["stride"], dc["umbral_score"],
                                dc["umbral_nms"], dc["max_det_por_clase"], tam)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_modelo = time.perf_counter() - t1

    # coherencia: si el clasificador dice que la region no esta en el corte, se apaga
    # su mascara y su caja (la cabeza que "mira toda la imagen" filtra falsos positivos)
    if cfg["postproceso"].get("filtrar_con_clasificador", True):
        presente = cls_p[:, :3] >= 0.5
        for r in range(3):
            sem[(~presente[:, r])[:, None, None] & (sem == r + 1)] = 0
        dets_256 = [[d for d in dz if presente[z, d["clase"]]] for z, dz in enumerate(dets_256)]

    t2 = time.perf_counter()
    esc = info["lado"] / tam
    pt, _, pl, _ = info["pad"]
    dets = [[{**d, "caja": [d["caja"][0] * esc - pl, d["caja"][1] * esc - pt,
                            d["caja"][2] * esc - pl, d["caja"][3] * esc - pt]} for d in dz] for dz in dets_256]
    sem_n = _a_nativo_por_bloques(sem, info, "nearest", np.uint8)
    rol_n = _a_nativo_por_bloques(rol, info, "nearest", np.uint8)
    borde_n = _a_nativo_por_bloques(borde.astype(np.float32), info, "bilinear", np.float16)
    dist_n = _a_nativo_por_bloques(dist.astype(np.float32), info, "bilinear", np.float16)
    pp = cfg["postproceso"]
    etq = volumen_etiquetas(sem_n, rol_n, borde_n.astype(np.float32), spacing,
                            pp["min_voxeles_fragmento"], pp["max_fragmentos_por_region"])
    tabla = tabla_fragmentos(etq, spacing, dist_n.astype(np.float32), pp["percentil_distancia"])
    t_post = time.perf_counter() - t2

    return {
        "etiquetas": etq, "detecciones": dets, "cls_prob": cls_p, "tabla": tabla,
        "borde": borde_n, "spacing": spacing,
        "latencia": {"dispositivo": device.type, "preproceso_s": t_pre, "modelo_s": t_modelo,
                     "postproceso_s": t_post, "ms_por_corte": 1000 * t_modelo / max(Z, 1), "n_cortes": Z},
    }
