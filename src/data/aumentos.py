"""Aumentos de datos.

Dos familias:
1. Geometricos SINCRONIZADOS: la misma transformacion a imagen, etiqueta y mapa de
   distancia. Imagen con interpolacion bilineal; etiqueta y distancia con vecino mas
   cercano (una interpolacion bilineal de la etiqueta inventaria ids que no existen).
2. Degradacion de calidad: solo a la imagen. Es la respuesta directa al pedido del
   profesor de que el modelo saque informacion de imagenes "feas": si nunca ve cortes
   ruidosos, borrosos o de baja resolucion en entrenamiento, no puede ser robusto a ellos.
   Los mismos operadores, con niveles FIJOS, generan el subconjunto de prueba degradado.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ degradacion
def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.05:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[0]
    x4 = x.unsqueeze(0)
    x4 = F.conv2d(F.pad(x4, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x4 = F.conv2d(F.pad(x4, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x4.squeeze(0)


def degradar(x: torch.Tensor, ruido=0.0, blur=0.0, submuestreo=1.0, gamma=1.0) -> torch.Tensor:
    """x: (C, H, W) en [0, 1]."""
    h, w = x.shape[-2:]
    if submuestreo > 1.01:  # perdida de resolucion efectiva (cortes gruesos / escaner viejo)
        hh, ww = max(8, int(h / submuestreo)), max(8, int(w / submuestreo))
        x = F.interpolate(x.unsqueeze(0), size=(hh, ww), mode="bilinear", align_corners=False, antialias=True)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    x = _blur(x, blur)
    if abs(gamma - 1.0) > 1e-3:
        x = x.clamp(0, 1) ** gamma
    if ruido > 0:
        x = x + torch.randn_like(x) * ruido
    return x.clamp(0, 1)


NIVELES_DEGRADACION = {
    # niveles fijos para el subconjunto de prueba degradado (se reportan en el informe)
    0: dict(ruido=0.0, blur=0.0, submuestreo=1.0, gamma=1.0),
    1: dict(ruido=0.03, blur=0.7, submuestreo=1.5, gamma=1.0),
    2: dict(ruido=0.06, blur=1.2, submuestreo=2.0, gamma=1.2),
    3: dict(ruido=0.10, blur=1.8, submuestreo=3.0, gamma=1.4),
}


def degradacion_aleatoria(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    u = random.uniform
    return degradar(
        x,
        ruido=u(*cfg["ruido_sigma"]),
        blur=u(*cfg["blur_sigma"]),
        submuestreo=u(*cfg["factor_submuestreo"]),
        gamma=u(*cfg["gamma"]),
    )


# ------------------------------------------------------------------ geometricos
def _swap_lateralidad(lbl: torch.Tensor) -> torch.Tensor:
    """Al voltear horizontalmente, el coxal izquierdo pasa a verse como derecho:
    hay que intercambiar 11..20 <-> 21..30 o se ensena la lateralidad al reves."""
    out = lbl.clone()
    izq = (lbl >= 11) & (lbl <= 20)
    der = (lbl >= 21) & (lbl <= 30)
    out[izq] = lbl[izq] + 10
    out[der] = lbl[der] - 10
    return out


def geometrico(img, lbl, dist, grados=10.0, escala=(0.9, 1.1), flip=False):
    """Rotacion + escala + (opcional) flip horizontal con swap de lateralidad."""
    if flip and random.random() < 0.5:
        img, lbl, dist = img.flip(-1), _swap_lateralidad(lbl.flip(-1)), dist.flip(-1)
    ang = math.radians(random.uniform(-grados, grados))
    s = random.uniform(*escala)
    cos, sin = math.cos(ang) / s, math.sin(ang) / s
    theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=torch.float32).unsqueeze(0)
    h, w = img.shape[-2:]
    grid = F.affine_grid(theta, (1, 1, h, w), align_corners=False)
    img = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    lbl = F.grid_sample(lbl[None, None].float(), grid, mode="nearest", align_corners=False)[0, 0].to(lbl.dtype)
    dist = F.grid_sample(dist[None, None].float(), grid, mode="nearest", align_corners=False)[0, 0]
    return img, lbl, dist
