"""Bucle de entrenamiento y validacion (lo usan scripts/train.py y scripts/overfit_test.py).

Incluye lo que el curso ya valido en Segmentacion S1 / Taller 3:
  * AMP (torch.autocast + GradScaler: la API actual de torch.cuda.amp)
  * acumulacion de gradiente: batch efectivo mayor sin subir el pico de VRAM (4 GB)
  * clip de gradiente
  * mejor checkpoint por metrica de VALIDACION, no el ultimo (leccion del Taller 3)
  * checkpoint con huella SHA-256 de las predicciones sobre un lote fijo
  * fast_dev_run para depurar todo el pipeline en segundos
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from .eval.metricas_det import resumen_deteccion
from .losses.perdidas import PerdidaPengwin
from .postprocess.nms import detecciones
from .utils import huella_tensor


def mover(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def entrenar_epoca(modelo, loader, perdida: PerdidaPengwin, opt, scaler, device, cabezas, cfg_e, fast=False):
    modelo.train()
    acum = int(cfg_e["acumulacion"])
    usar_amp = bool(cfg_e["amp"]) and device.type == "cuda"
    suma, n = {}, 0
    opt.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        batch = mover(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp):
            out = modelo(batch["img"], cabezas)
        total, comp = perdida(out, batch, cabezas)      # la perdida se calcula en float32
        scaler.scale(total / acum).backward()
        if (i + 1) % acum == 0 or (i + 1) == len(loader):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], cfg_e["clip_grad"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        for k, v in comp.items():
            suma[k] = suma.get(k, 0.0) + v
        suma["total"] = suma.get("total", 0.0) + float(total.detach())
        n += 1
        if fast and i >= 1:
            break
    return {k: v / max(n, 1) for k, v in suma.items()}


@torch.no_grad()
def validar(modelo, loader, perdida: PerdidaPengwin, device, cabezas, cfg, fast=False) -> dict:
    """Perdidas + metricas rapidas a nivel de corte (las metricas 3D finales van en evaluar.py)."""
    modelo.eval()
    usar_amp = bool(cfg["entrenamiento"]["amp"]) and device.type == "cuda"
    suma, n = {}, 0
    y_cls, p_cls, preds, gts = [], [], [], []
    inter = np.zeros(4)
    union = np.zeros(4)
    dc = cfg["deteccion"]
    for i, batch in enumerate(loader):
        batch = mover(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp):
            out = modelo(batch["img"], cabezas)
        total, comp = perdida(out, batch, cabezas)
        for k, v in comp.items():
            suma[k] = suma.get(k, 0.0) + v
        suma["total"] = suma.get("total", 0.0) + float(total)
        n += 1
        if "cls" in out:
            y_cls.append(batch["cls"][:, :3].cpu().numpy())
            p_cls.append(out["cls"][:, :3].float().sigmoid().cpu().numpy())
        if "det_cls" in out:
            preds += detecciones(out["det_cls"], out["det_box"], dc["stride"], 0.05, dc["umbral_nms"],
                                 dc["max_det_por_clase"], cfg["datos"]["resolucion"])
            for b in range(batch["cajas"].shape[0]):
                gts.append([{"clase": r, "caja": batch["cajas"][b, r].tolist()}
                            for r in range(3) if bool(batch["hay_caja"][b, r])])
        if "sem" in out:
            pred = out["sem"].argmax(1)
            for c in range(1, 4):
                p, g = pred == c, batch["sem"] == c
                inter[c] += float((p & g).sum())
                union[c] += float(p.sum() + g.sum())
        if fast and i >= 1:
            break

    res = {f"val_{k}": v / max(n, 1) for k, v in suma.items()}
    partes = []
    if y_cls:
        from sklearn.metrics import f1_score
        y, p = np.concatenate(y_cls), np.concatenate(p_cls)
        res["val_f1_cls"] = float(f1_score(y, p >= 0.5, average="macro", zero_division=0))
        partes.append(res["val_f1_cls"])
    if preds:
        rd = resumen_deteccion(preds, gts)
        res["val_map50"] = rd["mAP@0.50"]
        res["val_iou_det"] = rd["iou_promedio"]
        partes.append(rd["mAP@0.50"])
    if union[1:].sum() > 0:
        dice = [2 * inter[c] / union[c] for c in range(1, 4) if union[c] > 0]
        res["val_dice_sem"] = float(np.mean(dice))
        partes.append(res["val_dice_sem"])
    res["val_score"] = float(np.nanmean(partes)) if partes else -res["val_total"]
    return res


def guardar_checkpoint(ruta, modelo, opt, scaler, etapa, epoca, cfg, metricas, lote_fijo=None, extra=None):
    ck = {"modelo": modelo.state_dict(), "optimizador": opt.state_dict(), "scaler": scaler.state_dict(),
          "etapa": etapa, "epoca": epoca, "config": cfg, "metricas": metricas, "fecha": time.strftime("%Y-%m-%d %H:%M")}
    if lote_fijo is not None:
        modelo.eval()
        with torch.no_grad():
            ck["huella_predicciones"] = huella_tensor(modelo(lote_fijo, ("cls",))["cls"])
    if extra:
        ck.update(extra)
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, ruta)


CAMPOS_LOG = ["etapa", "epoca", "lr", "seg_epoca", "total", "cls", "det_obj", "det_box", "sem", "rol",
              "borde", "dist", "val_total", "val_cls", "val_det_obj", "val_det_box", "val_sem", "val_rol",
              "val_borde", "val_dist", "val_f1_cls", "val_map50", "val_iou_det", "val_dice_sem", "val_score"]


class RegistroCSV:
    def __init__(self, ruta, campos=CAMPOS_LOG):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.campos = list(campos)

    def escribir(self, fila: dict):
        nuevo = not self.ruta.exists()
        with open(self.ruta, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.campos, extrasaction="ignore")
            if nuevo:
                w.writeheader()
            w.writerow(fila)
