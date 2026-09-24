"""FundidoraPC extendida — backbone del proyecto.

Punto de partida: la FundidoraPC del curso (S6/S7/Taller 3): bloques
conv3x3 -> BatchNorm -> ReLU -> MaxPool con 32 -> 64 -> 128 -> 256 canales.

Extension para PENGWIN:
  * 5to bloque (512 canales): mas capacidad para 3 regiones + fragmentos.
  * `forward_features` devuelve la piramide de mapas ANTES de cada MaxPool
    (strides 1, 2, 4, 8, 16) para colgar el decoder U-Net con skip connections.
    Justificacion (Segmentacion S1): segmentar solo desde el mapa final a stride 16/32
    tiene un techo de IoU por debajo de la meta de Dice >= 0.85.
  * CBAM en los bloques indicados (por defecto 4 y 5), antes de bifurcar a las cabezas.
  * `forward` conserva la firma del Taller 3: (pooled, feature_map).

Con entrada 256x256: f5 es 512 x 16 x 16  -> grid de deteccion de 16x16 celdas.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cbam import CBAM


def bloque_conv(c_in: int, c_out: int, n_convs: int = 1) -> nn.Sequential:
    capas = []
    for i in range(n_convs):
        capas += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False),
                  nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*capas)


class FundidoraPCExtendida(nn.Module):
    def __init__(self, in_ch=3, canales=(32, 64, 128, 256, 512), convs_por_bloque=1,
                 cbam_en_bloques=(4, 5), cbam_reduccion=16):
        super().__init__()
        self.canales = list(canales)
        self.bloques = nn.ModuleList()
        c_prev = in_ch
        for c in canales:
            self.bloques.append(bloque_conv(c_prev, c, convs_por_bloque))
            c_prev = c
        self.cbam = nn.ModuleDict({
            str(b): CBAM(canales[b - 1], cbam_reduccion) for b in cbam_en_bloques
        })
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward_features(self, x) -> list[torch.Tensor]:
        feats = []
        for i, bloque in enumerate(self.bloques, start=1):
            x = bloque(x)
            if str(i) in self.cbam:
                x = self.cbam[str(i)](x)
            feats.append(x)            # mapa ANTES del pool (skip para el decoder)
            if i < len(self.bloques):
                x = self.pool(x)
        return feats

    def forward(self, x):
        feats = self.forward_features(x)
        pooled = self.gap(self.pool(feats[-1])).flatten(1)
        return pooled, feats[-1]


def cargar_pesos_parciales(modelo: nn.Module, ruta: str, verbose=True) -> int:
    """Transfer learning en el backbone (unico lugar donde el enunciado lo permite).

    Copia, EN ORDEN, los pesos de conv/BN de una FundidoraPC entrenada antes en el curso
    (Intel Images, Face Mask, etc.) a los bloques equivalentes de la extendida. Se empareja
    por orden y forma, no por nombre, porque cada notebook nombro las capas distinto.
    Si la primera conv se entreno con 1 canal (FashionMNIST) se replica a 3 canales.
    Devuelve cuantos tensores se copiaron (reportarlo en el model card).
    """
    fuente = torch.load(ruta, map_location="cpu")
    if isinstance(fuente, dict) and "modelo" in fuente:
        fuente = fuente["modelo"]
    if isinstance(fuente, nn.Module):
        fuente = fuente.state_dict()

    def claves_conv_bn(sd):
        return [k for k, v in sd.items()
                if any(t in k for t in ("weight", "bias", "running_mean", "running_var"))
                and "cbam" not in k and "fc" not in k and "head" not in k and "cabeza" not in k
                and v.ndim in (1, 4)]

    destino = modelo.state_dict()
    k_src, k_dst = claves_conv_bn(fuente), [k for k in claves_conv_bn(destino) if k.startswith("bloques")]
    copiados = 0
    for ks, kd in zip(k_src, k_dst):
        ws, wd = fuente[ks], destino[kd]
        if ws.shape == wd.shape:
            destino[kd] = ws.clone()
            copiados += 1
        elif ws.ndim == 4 and ws.shape[0] == wd.shape[0] and ws.shape[2:] == wd.shape[2:]:
            destino[kd] = ws.mean(1, keepdim=True).repeat(1, wd.shape[1], 1, 1)  # 1 -> 3 canales
            copiados += 1
        else:
            break  # a partir de aqui la arquitectura diverge (bloque 5 nuevo)
    modelo.load_state_dict(destino)
    if verbose:
        print(f"[transfer] {copiados} tensores copiados desde {ruta}")
    return copiados
