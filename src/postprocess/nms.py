"""NMS implementado por el equipo (requisito 3.1) — misma logica de nms_manual del curso.

Por clase: ordenar por score, quedarse con la mejor caja, descartar las que tengan
IoU > umbral con ella, repetir con las restantes. Validar contra torchvision.ops.nms
en tests/ (solo como verificacion; el pipeline usa esta implementacion).
"""
from __future__ import annotations

import torch

from ..losses.perdidas import decodificar_cajas


def iou_matriz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a (N,4), b (M,4) x1y1x2y2 -> IoU (N,M) vectorizado."""
    ix1 = torch.max(a[:, None, 0], b[None, :, 0])
    iy1 = torch.max(a[:, None, 1], b[None, :, 1])
    ix2 = torch.min(a[:, None, 2], b[None, :, 2])
    iy2 = torch.min(a[:, None, 3], b[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).clamp(0)
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).clamp(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def nms_manual(cajas: torch.Tensor, scores: torch.Tensor, umbral: float = 0.5) -> torch.Tensor:
    """Devuelve los indices que sobreviven, ordenados por score descendente."""
    orden = scores.argsort(descending=True)
    keep = []
    while orden.numel() > 0:
        i = orden[0]
        keep.append(int(i))
        if orden.numel() == 1:
            break
        ious = iou_matriz(cajas[i].unsqueeze(0), cajas[orden[1:]])[0]
        orden = orden[1:][ious <= umbral]
    return torch.tensor(keep, dtype=torch.long, device=cajas.device)


def detecciones(det_cls: torch.Tensor, det_box: torch.Tensor, stride=16, umbral_score=0.3,
                umbral_nms=0.5, max_por_clase=1, tam_img=256) -> list[list[dict]]:
    """Salida cruda de la cabeza -> lista (por imagen) de detecciones
    {'clase': 0..2, 'score': float, 'caja': [x1,y1,x2,y2]}. NMS por clase."""
    cajas = decodificar_cajas(det_box.float(), stride).clamp(0, tam_img)   # (B,4,h,w)
    probs = det_cls.float().sigmoid()                                       # (B,3,h,w)
    B, C = probs.shape[:2]
    salida = []
    for b in range(B):
        cb = cajas[b].reshape(4, -1).T
        dets = []
        for c in range(C):
            s = probs[b, c].reshape(-1)
            m = s >= umbral_score
            if not m.any():
                continue
            keep = nms_manual(cb[m], s[m], umbral_nms)[:max_por_clase]
            for k in keep:
                dets.append({"clase": c, "score": float(s[m][k]), "caja": cb[m][k].tolist()})
        salida.append(dets)
    return salida
