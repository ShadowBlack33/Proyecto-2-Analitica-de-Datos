"""Dataset 2.5D corte a corte.

Cada muestra es un corte axial con `contexto` cortes vecinos a cada lado como canales
(3 canales con contexto=1). Asi el modelo 2D ve algo de la estructura 3D local — el
"fundamento del analisis 3D" del enunciado — sin el costo de memoria de una red 3D, y
los 3 canales son compatibles con pesos preentrenados en imagenes RGB.

Targets que se generan por corte (todos a 256x256):
    sem   (H,W)  long   region 0 fondo / 1 SA / 2 LI / 3 RI
    rol   (H,W)  long   0 fondo / 1 fragmento principal / 2 fragmento secundario
    borde (H,W)  float  superficie de contacto entre fragmentos
    dist  (H,W)  float  mm al fragmento principal (solo valido donde rol == 2)
    cls   (6,)   float  [presencia SA, LI, RI, fractura SA, LI, RI]
    cajas (3,4)  float  caja de cada region en pixeles x1,y1,x2,y2 (ceros si no esta)
    hay_caja (3,) bool
"""
from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import aumentos as A
from .etiquetas import indice_en_region, mapa_borde, region_de_etiqueta


class CacheCaso:
    """Acceso perezoso (mmap) a los .npy de un caso."""

    def __init__(self, dir_caso: Path):
        self.dir = Path(dir_caso)
        with open(self.dir / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self._img = self._lbl = self._dist = None

    @property
    def img(self):
        if self._img is None:
            self._img = np.load(self.dir / "img.npy", mmap_mode="r")
        return self._img

    @property
    def lbl(self):
        if self._lbl is None:
            self._lbl = np.load(self.dir / "lbl.npy", mmap_mode="r")
        return self._lbl

    @property
    def dist(self):
        if self._dist is None:
            self._dist = np.load(self.dir / "dist.npy", mmap_mode="r")
        return self._dist

    @property
    def n_cortes(self) -> int:
        return int(self.img.shape[0])


def pila_contexto(img_vol, z: int, contexto: int) -> np.ndarray:
    """Cortes z-c..z+c (repitiendo el borde del volumen)."""
    Z = img_vol.shape[0]
    idx = [min(max(z + k, 0), Z - 1) for k in range(-contexto, contexto + 1)]
    return np.stack([img_vol[i] for i in idx], 0)


def construir_targets(lbl: torch.Tensor, dist: torch.Tensor) -> dict:
    lbl_np = lbl.numpy().astype(np.uint8)
    sem = region_de_etiqueta(lbl_np)
    idx = indice_en_region(lbl_np)
    rol = np.zeros_like(idx)
    rol[idx == 1] = 1
    rol[idx >= 2] = 2
    borde = mapa_borde(lbl_np)

    cls = np.zeros(6, np.float32)
    cajas = np.zeros((3, 4), np.float32)
    hay = np.zeros(3, bool)
    for r in (1, 2, 3):
        m = sem == r
        if m.any():
            ys, xs = np.where(m)
            cajas[r - 1] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            hay[r - 1] = True
            cls[r - 1] = 1.0
            cls[3 + r - 1] = float((m & (idx >= 2)).any())
    return {
        "sem": torch.from_numpy(sem.astype(np.int64)),
        "rol": torch.from_numpy(rol.astype(np.int64)),
        "borde": torch.from_numpy(borde.astype(np.float32)),
        "dist": dist.float() * torch.from_numpy(rol == 2).float(),
        "cls": torch.from_numpy(cls),
        "cajas": torch.from_numpy(cajas),
        "hay_caja": torch.from_numpy(hay),
    }


class PengwinCortes(Dataset):
    def __init__(self, dir_cache, ids, cfg, entrenamiento=False, nivel_degradacion=0, seed=42):
        self.cfg = cfg
        self.ctx = int(cfg["datos"]["contexto"])
        self.entrenamiento = entrenamiento
        self.nivel_degradacion = nivel_degradacion
        self.casos = {cid: CacheCaso(Path(dir_cache) / cid) for cid in ids}
        rng = random.Random(seed)
        frac_fondo = cfg["datos"]["frac_cortes_sin_hueso"]
        self.indice = []
        for cid, c in self.casos.items():
            lbl = c.lbl
            con_hueso = lbl.reshape(lbl.shape[0], -1).max(1) > 0
            for z in range(c.n_cortes):
                # en evaluacion se usan todos los cortes; en entrenamiento se submuestrea el fondo
                if con_hueso[z] or not entrenamiento or rng.random() < frac_fondo:
                    self.indice.append((cid, z))

    def __len__(self):
        return len(self.indice)

    def __getitem__(self, i):
        cid, z = self.indice[i]
        c = self.casos[cid]
        img = torch.from_numpy(pila_contexto(c.img, z, self.ctx).astype(np.float32) / 255.0)
        lbl = torch.from_numpy(np.array(c.lbl[z], dtype=np.int64))
        dist = torch.from_numpy(np.array(c.dist[z], dtype=np.float32))

        au = self.cfg["aumentos"]
        if self.entrenamiento and au["activar"]:
            img, lbl, dist = A.geometrico(img, lbl, dist, grados=au["rotacion_grados"], flip=au["flip_horizontal"])
            if random.random() < au["prob_degradacion"]:
                img = A.degradacion_aleatoria(img, au)
        elif self.nivel_degradacion:
            # semilla fija por corte: el subconjunto degradado es identico en cada evaluacion
            torch.manual_seed(zlib.crc32(f"{cid}-{z}".encode()))
            img = A.degradar(img, **A.NIVELES_DEGRADACION[self.nivel_degradacion])

        t = construir_targets(lbl, dist)
        t["img"] = img
        t["caso"] = cid
        t["z"] = z
        return t
