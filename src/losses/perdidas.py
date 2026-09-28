"""Perdida compuesta multitarea.

    L = l_cls*L_cls + l_obj*L_det_obj + l_box*L_det_box
      + l_sem*L_sem + l_rol*L_rol + l_borde*L_borde + l_dist*L_dist

Los lambdas NO se copian de otras implementaciones: scripts/calibrar_lambdas.py mide la
magnitud de cada termino y la norma de su gradiente sobre el backbone compartido al
inicio del entrenamiento (la misma leccion de S6: sin ponderar, L_cls dominaba ~45x
a L_box) y propone lambdas que equilibran gradientes. El informe cita esa medicion.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ============================================================ deteccion: targets
def centros_grid(h: int, w: int, stride: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
    cy, cx = torch.meshgrid(ys, xs, indexing="ij")
    return cx, cy


def asignar_targets(cajas, hay_caja, h, w, stride=16, radio=2.5):
    """Asignacion anchor-free con center sampling (FCOS / YOLOX).

    cajas (B,3,4) en pixeles x1y1x2y2, hay_caja (B,3).
    Una celda es positiva para la region r si su centro cae dentro de la caja de r y a
    menos de `radio` celdas del centro de la caja. Si una caja es tan pequena que ninguna
    celda cumple, se fuerza la celda que contiene su centro (regla de rescate, como la
    del Taller 3). Si dos cajas reclaman la misma celda, gana la de menor area.
    Devuelve: cls_t (B,3,h,w), ltrb_t (B,4,h,w) en unidades de celda, pos (B,h,w), caja_t (B,4,h,w).
    """
    B = cajas.shape[0]
    dev = cajas.device
    cx, cy = centros_grid(h, w, stride, dev)
    cls_t = torch.zeros(B, 3, h, w, device=dev)
    ltrb_t = torch.zeros(B, 4, h, w, device=dev)
    caja_t = torch.zeros(B, 4, h, w, device=dev)
    area_asig = torch.full((B, h, w), float("inf"), device=dev)
    for b in range(B):
        for r in range(3):
            if not bool(hay_caja[b, r]):
                continue
            x1, y1, x2, y2 = cajas[b, r]
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            dentro = (cx > x1) & (cx < x2) & (cy > y1) & (cy < y2)
            cerca = ((cx - bcx).abs() < radio * stride) & ((cy - bcy).abs() < radio * stride)
            m = dentro & cerca
            if not m.any():
                j = int(torch.clamp(bcx // stride, 0, w - 1))
                i = int(torch.clamp(bcy // stride, 0, h - 1))
                m = torch.zeros_like(dentro)
                m[i, j] = True
            area = (x2 - x1) * (y2 - y1)
            m = m & (area < area_asig[b])
            area_asig[b][m] = area
            cls_t[b, :, m] = 0.0
            cls_t[b, r, m] = 1.0
            ltrb = torch.stack([cx - x1, cy - y1, x2 - cx, y2 - cy]) / stride
            ltrb_t[b][:, m] = ltrb.clamp(min=0)[:, m]
            caja_t[b][:, m] = torch.stack([x1, y1, x2, y2]).view(4, 1).expand(4, int(m.sum()))
    pos = cls_t.sum(1) > 0
    return cls_t, ltrb_t, pos, caja_t


def decodificar_cajas(ltrb, stride=16):
    """ltrb (B,4,h,w) en celdas -> cajas (B,4,h,w) en pixeles x1y1x2y2."""
    B, _, h, w = ltrb.shape
    cx, cy = centros_grid(h, w, stride, ltrb.device)
    l, t, r, b = (ltrb * stride).unbind(1)
    return torch.stack([cx - l, cy - t, cx + r, cy + b], dim=1)


def giou(a, b, eps=1e-7):
    """a, b (N,4) x1y1x2y2 -> GIoU (N,)."""
    ix1, iy1 = torch.max(a[:, 0], b[:, 0]), torch.max(a[:, 1], b[:, 1])
    ix2, iy2 = torch.min(a[:, 2], b[:, 2]), torch.min(a[:, 3], b[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    union = area_a + area_b - inter + eps
    iou = inter / union
    cx1, cy1 = torch.min(a[:, 0], b[:, 0]), torch.min(a[:, 1], b[:, 1])
    cx2, cy2 = torch.max(a[:, 2], b[:, 2]), torch.max(a[:, 3], b[:, 3])
    c = (cx2 - cx1) * (cy2 - cy1) + eps
    return iou - (c - union) / c


def focal_sigmoid(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return a_t * (1 - p_t) ** gamma * ce


# ============================================================ segmentacion
def dice_multiclase(logits, target, clases, eps=1.0):
    """Dice suave promedio sobre `clases` (sin el fondo)."""
    prob = logits.float().softmax(1)
    oh = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * oh).sum(dims)
    den = prob.sum(dims) + oh.sum(dims)
    dice = (2 * inter + eps) / (den + eps)
    return 1 - dice[clases].mean()


# ============================================================ total
class PerdidaPengwin:
    def __init__(self, cfg: dict):
        self.c = cfg["perdida"]
        self.stride = cfg["deteccion"]["stride"]
        self.radio = cfg["deteccion"]["radio_centro"]

    def __call__(self, out: dict, batch: dict, cabezas) -> tuple[torch.Tensor, dict]:
        c = self.c
        comp = {}

        if "cls" in cabezas:
            comp["cls"] = F.binary_cross_entropy_with_logits(out["cls"].float(), batch["cls"])

        if "det" in cabezas:
            dc, db = out["det_cls"].float(), out["det_box"].float()
            h, w = dc.shape[-2:]
            cls_t, ltrb_t, pos, caja_t = asignar_targets(batch["cajas"], batch["hay_caja"], h, w,
                                                         self.stride, self.radio)
            n_pos = max(1.0, float(pos.sum()))
            comp["det_obj"] = focal_sigmoid(dc, cls_t, c["focal_alpha"], c["focal_gamma"]).sum() / n_pos
            if pos.any():
                pred = decodificar_cajas(db, self.stride).permute(0, 2, 3, 1)[pos]
                gt = caja_t.permute(0, 2, 3, 1)[pos]
                comp["det_box"] = (1 - giou(pred, gt)).mean()
            else:
                comp["det_box"] = db.sum() * 0.0

        if "seg" in cabezas:
            sem, rol, borde = out["sem"].float(), out["rol"].float(), out["borde"].float()
            comp["sem"] = F.cross_entropy(sem, batch["sem"]) + dice_multiclase(sem, batch["sem"], [1, 2, 3])
            comp["rol"] = F.cross_entropy(rol, batch["rol"]) + dice_multiclase(rol, batch["rol"], [1, 2])
            pw = torch.tensor(c["borde_pos_weight"], device=borde.device)
            comp["borde"] = F.binary_cross_entropy_with_logits(borde[:, 0], batch["borde"], pos_weight=pw)

        if "dist" in cabezas:
            m = batch["rol"] == 2
            pred = out["dist"][:, 0].float()
            if m.any():
                # log(1+d): con un rango de 2 a 250 mm, los pocos casos extremos
                # no dominan el gradiente; Huber ademas limita el efecto de outliers
                comp["dist"] = F.huber_loss(pred[m], torch.log1p(batch["dist"][m]), delta=c["huber_delta"])
            else:
                comp["dist"] = pred.sum() * 0.0

        pesos = {"cls": c["lambda_cls"], "det_obj": c["lambda_det_obj"], "det_box": c["lambda_det_box"],
                 "sem": c["lambda_sem"], "rol": c["lambda_rol"], "borde": c["lambda_borde"],
                 "dist": c["lambda_dist"]}
        total = sum(pesos[k] * v for k, v in comp.items())
        return total, {k: float(v.detach()) for k, v in comp.items()}
