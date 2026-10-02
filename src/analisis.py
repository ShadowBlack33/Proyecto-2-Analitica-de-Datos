"""Analisis de interpretabilidad del modelo, para el notebook y el informe.

1. gamma de BatchNorm por bloque: que tanto usa la red cada capa.
2. Grad-CAM (y Smooth Grad-CAM): que mira la cabeza de clasificacion al decidir cada hueso,
   comparado con el mapa de objectness de la cabeza de deteccion (mismo analisis del Taller 3).
3. Cajas: candidatos antes y despues del NMS, e IoU contra el ground truth en test.
4. Casos con fractura: elegir pacientes y cortes donde si hay fragmentos secundarios.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data.etiquetas import COLORES_REGION, REGIONES, indice_en_region, region_de_etiqueta
from .losses.perdidas import decodificar_cajas
from .models.modelo import PengwinNet
from .postprocess.nms import detecciones, iou_matriz


# ============================================================ 1. gamma de BatchNorm
def _nombre_capa(prefijo: str) -> str:
    p = prefijo.split(".")
    if p[0] == "backbone" and p[1] == "bloques":
        return f"Backbone · bloque {int(p[2]) + 1}"
    if p[0] == "decoder" and p[1] == "ups":
        return f"Decoder · subida {int(p[2]) + 1} · BN {'a' if p[4] == '1' else 'b'}"
    return prefijo


def gammas_batchnorm(estado: dict, umbral_relativo: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Estadisticas del gamma (escala aprendida) de cada BatchNorm de un state_dict.

    Cada canal se normaliza y luego se multiplica por su gamma: si gamma ~ 0 el canal casi
    no llega a la capa siguiente. 'pct_apagados' es el % de canales con |gamma| menor que
    `umbral_relativo` x el maximo |gamma| de esa misma capa (criterio usado para poda).
    """
    filas, crudos = [], {}
    for k in estado:
        if not k.endswith(".running_mean"):
            continue
        pref = k[: -len(".running_mean")]
        g = estado[pref + ".weight"].detach().float().cpu().numpy()
        b = estado[pref + ".bias"].detach().float().cpu().numpy()
        a = np.abs(g)
        nombre = _nombre_capa(pref)
        crudos[nombre] = g
        filas.append({"capa": nombre, "modulo": pref, "canales": len(g),
                      "gamma_medio_abs": a.mean(), "gamma_mediana_abs": np.median(a),
                      "gamma_min_abs": a.min(), "gamma_max_abs": a.max(),
                      "pct_apagados": 100 * (a < umbral_relativo * a.max()).mean(),
                      "beta_medio": b.mean()})
    return pd.DataFrame(filas), crudos


def cargar_estado(ruta) -> dict:
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    return ck["modelo"] if "modelo" in ck else ck


def modelo_desde_checkpoint(ruta, cfg_base: dict, device) -> PengwinNet:
    """Construye el modelo con la configuracion guardada en el checkpoint (sirve para las
    ablaciones, que tienen otra arquitectura, p. ej. sin CBAM)."""
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or cfg_base
    modelo = PengwinNet(cfg["modelo"], in_ch=2 * cfg["datos"]["contexto"] + 1)
    modelo.load_state_dict(ck["modelo"] if "modelo" in ck else ck)
    return modelo.to(device).eval()


# ============================================================ 2. Grad-CAM
def _normalizar(m: torch.Tensor) -> torch.Tensor:
    m = m - m.min()
    return m / (m.max() + 1e-8)


def grad_cam(modelo: PengwinNet, img: torch.Tensor, region: int, n_suave: int = 8, ruido: float = 0.1,
             seed: int = 0) -> np.ndarray:
    """Grad-CAM de la cabeza de clasificacion para la region 0 SA / 1 LI / 2 RI, sobre el
    ultimo mapa del backbone (f5, despues de CBAM). Con n_suave > 1 es Smooth Grad-CAM:
    promedia el mapa sobre copias de la entrada con ruido gaussiano."""
    dev = next(modelo.parameters()).device
    x = img.to(dev).float().unsqueeze(0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    mapas = []
    for k in range(max(1, n_suave)):
        xi = x if n_suave <= 1 else x + ruido * torch.randn(x.shape, generator=g).to(dev)
        with torch.no_grad():
            f5 = modelo.backbone.forward_features(xi)[-1]
        f5 = f5.detach().requires_grad_(True)
        with torch.enable_grad():
            logit = modelo.cabeza_cls(f5)[0, region]
            grad, = torch.autograd.grad(logit, f5)
        pesos = grad.mean((2, 3), keepdim=True)                  # importancia de cada canal
        cam = F.relu((pesos * f5).sum(1, keepdim=True))
        mapas.append(F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0])
    return _normalizar(torch.stack(mapas).mean(0)).detach().cpu().numpy()


@torch.no_grad()
def objectness(modelo: PengwinNet, img: torch.Tensor, region: int) -> tuple[np.ndarray, float]:
    """Mapa de objectness de la cabeza de deteccion (grid 16x16 llevado a la imagen) y la
    probabilidad de presencia que da la cabeza de clasificacion."""
    dev = next(modelo.parameters()).device
    x = img.to(dev).float().unsqueeze(0)
    out = modelo(x, ("cls", "det"))
    obj = out["det_cls"].float().sigmoid()[:, region:region + 1]
    obj = F.interpolate(obj, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return obj.cpu().numpy(), float(out["cls"].float().sigmoid()[0, region])


def masa_en_caja(mapa: np.ndarray, caja) -> float:
    """Fraccion del mapa que cae dentro de la caja [x1, y1, x2, y2] (1 = todo dentro)."""
    x1, y1, x2, y2 = [int(round(v)) for v in caja]
    total = mapa.sum()
    return float(mapa[y1:y2, x1:x2].sum() / total) if total > 0 else float("nan")


def comparar_gradcam_objectness(modelo, img, region, caja_gt, n_suave=8) -> dict:
    cam = grad_cam(modelo, img, region, n_suave=n_suave)
    obj, prob = objectness(modelo, img, region)
    corr = float(np.corrcoef(cam.ravel(), obj.ravel())[0, 1]) if cam.std() > 0 and obj.std() > 0 else float("nan")
    area = (caja_gt[2] - caja_gt[0]) * (caja_gt[3] - caja_gt[1]) / cam.size
    return {"cam": cam, "obj": obj, "prob_cls": prob, "correlacion": corr,
            "masa_cam": masa_en_caja(cam, caja_gt), "masa_obj": masa_en_caja(obj, caja_gt),
            "masa_uniforme": float(area)}


# ============================================================ 3. cajas
@torch.no_grad()
def cajas_de_corte(modelo, img, cfg) -> dict:
    """Candidatos de la cabeza de deteccion antes y despues del NMS para un corte."""
    dev = next(modelo.parameters()).device
    D = cfg["deteccion"]
    out = modelo(img.to(dev).float().unsqueeze(0), ("cls", "det"))
    tam = img.shape[-1]
    cajas = decodificar_cajas(out["det_box"].float(), D["stride"]).clamp(0, tam)[0]
    probs = out["det_cls"].float().sigmoid()[0]
    candidatos = []
    for c in range(probs.shape[0]):
        m = probs[c] >= D["umbral_score"]
        for caja, s in zip(cajas.permute(1, 2, 0)[m].tolist(), probs[c][m].tolist()):
            candidatos.append({"clase": c, "score": s, "caja": caja})
    finales = detecciones(out["det_cls"], out["det_box"], D["stride"], D["umbral_score"], D["umbral_nms"],
                          D["max_det_por_clase"], tam)[0]
    presencia = out["cls"].float().sigmoid()[0, :3].cpu().numpy()
    return {"candidatos": candidatos, "finales": finales, "presencia": presencia}


def iou_caja(a, b) -> float:
    return float(iou_matriz(torch.tensor([a], dtype=torch.float32), torch.tensor([b], dtype=torch.float32))[0, 0])


@torch.no_grad()
def iou_cajas_en_test(modelo, dataset, cfg, cada: int = 2, batch: int = 16) -> pd.DataFrame:
    """IoU de la caja final de cada region contra la del ground truth, corte por corte.
    Si la region esta en el corte y no se detecta, IoU = 0 ('perdida'). Si no esta y se
    detecta, cuenta como falso positivo."""
    from torch.utils.data import DataLoader, Subset

    dev = next(modelo.parameters()).device
    D = cfg["deteccion"]
    sub = Subset(dataset, list(range(0, len(dataset), cada)))
    filas = []
    for b in DataLoader(sub, batch, shuffle=False, num_workers=0):
        out = modelo(b["img"].to(dev).float(), ("cls", "det"))
        dets = detecciones(out["det_cls"], out["det_box"], D["stride"], D["umbral_score"], D["umbral_nms"],
                           D["max_det_por_clase"], b["img"].shape[-1])
        pres = out["cls"].float().sigmoid()[:, :3].cpu().numpy() >= 0.5
        for i in range(len(dets)):
            pred = {d["clase"]: d for d in dets[i] if pres[i, d["clase"]]}
            for r in range(3):
                hay = bool(b["hay_caja"][i, r])
                if not hay and r not in pred:
                    continue
                fila = {"caso": b["caso"][i], "z": int(b["z"][i]), "region": REGIONES[r], "en_gt": hay,
                        "detectada": r in pred, "score": pred[r]["score"] if r in pred else float("nan"), "iou": 0.0}
                if hay and r in pred:
                    fila["iou"] = iou_caja(pred[r]["caja"], b["cajas"][i, r].tolist())
                filas.append(fila)
    return pd.DataFrame(filas)


# ============================================================ 4. casos con fractura
def cortes_con_fractura(dir_cache, ids) -> pd.DataFrame:
    """Por paciente: cuantos fragmentos secundarios tiene y el corte con mas area de ellos.
    'z_nativo' es el indice del corte en el volumen original (para leer_volumen)."""
    filas = []
    for cid in ids:
        d = Path(dir_cache) / cid
        lbl = np.load(d / "lbl.npy", mmap_mode="r")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        idx = indice_en_region(np.asarray(lbl))
        sec = idx >= 2
        area = sec.reshape(sec.shape[0], -1).sum(1)
        etiquetas = [int(v) for v in np.unique(np.asarray(lbl)[sec])]
        z = int(area.argmax())
        filas.append({"caso": cid, "secundarios": len(etiquetas),
                      "regiones_fracturadas": len({int(region_de_etiqueta(v)) for v in etiquetas}),
                      "cortes_con_fractura": int((area > 0).sum()), "area_max_px": int(area.max()),
                      "z_cache": z, "z_nativo": int(meta.get("z0", 0)) + z})
    return pd.DataFrame(filas).sort_values(["secundarios", "area_max_px"], ascending=False).reset_index(drop=True)


def colorear_etiquetas(etq2d: np.ndarray) -> np.ndarray:
    """RGBA por fragmento: color del hueso; el principal oscuro y los secundarios en tonos
    cada vez mas claros, para distinguirlos entre si."""
    rgba = np.zeros(etq2d.shape + (4,), np.float32)
    for v in np.unique(etq2d):
        if v == 0:
            continue
        base = np.array(COLORES_REGION[REGIONES[int(region_de_etiqueta(v)) - 1]], np.float32) / 255
        i = int(indice_en_region(v))
        col = base * 0.75 if i == 1 else base + (1 - base) * min(0.25 + 0.18 * (i - 2), 0.85)
        rgba[etq2d == v] = (*col, 0.75)
    return rgba
