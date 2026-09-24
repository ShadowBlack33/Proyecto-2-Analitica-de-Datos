"""CBAM — Convolutional Block Attention Module (Woo et al., ECCV 2018).

Dos atenciones en serie sobre un mapa F (B, C, H, W):
  1. Canal:    Mc = sigma( MLP(AvgPool(F)) + MLP(MaxPool(F)) )  -> (B, C, 1, 1)
               "QUE canales (filtros) importan": p.ej. los que responden a hueso cortical.
  2. Espacial: Ms = sigma( conv7x7([Avg_c(F'); Max_c(F')]) )    -> (B, 1, H, W)
               "DONDE mirar": resalta la zona de la pelvis y apaga ruido y fondo.
  F'' = Ms * (Mc * F)

Argumento para el informe: en cortes degradados (ruido, blur) la atencion espacial
aprende a ignorar regiones poco informativas; el ablation con/sin CBAM lo verifica.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AtencionCanal(nn.Module):
    def __init__(self, c: int, reduccion: int = 16):
        super().__init__()
        oculto = max(4, c // reduccion)
        self.mlp = nn.Sequential(nn.Conv2d(c, oculto, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(oculto, c, 1, bias=False))

    def forward(self, x):
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + mx)


class AtencionEspacial(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, c: int, reduccion: int = 16, k: int = 7):
        super().__init__()
        self.canal = AtencionCanal(c, reduccion)
        self.espacial = AtencionEspacial(k)
        self.ultimo_mapa_espacial = None  # para visualizarlo en el dashboard / informe

    def forward(self, x):
        x = x * self.canal(x)
        ms = self.espacial(x)
        self.ultimo_mapa_espacial = ms.detach()
        return x * ms
