"""PengwinNet: un backbone compartido (FundidoraPC extendida + CBAM) y 4 cabezas.

  cls   clasificacion anatomica del corte (mira TODA la imagen):
        6 logits = presencia de SA/LI/RI + "tiene fragmentos secundarios" SA/LI/RI.
  det   deteccion anchor-free sobre un grid propio de 16x16 (stride 16):
        por celda, 3 logits de clase + 4 distancias a los bordes (l, t, r, b) — la idea de
        FCOS/YOLOv8, implementada a mano (sin anclas que disenar). NMS propio aparte.
  seg   decoder U-Net con skips -> 3 salidas densas:
        sem (4: fondo/SA/LI/RI), rol (3: fondo/principal/secundario), borde (1: contacto).
        Semantica + instancia en dos etapas, como pide la seccion 3.2.
  dist  regresion densa: log(1 + mm al fragmento principal), valida en pixeles secundarios.
        La distancia de un fragmento = minimo del mapa dentro de el (post-proceso).

Las cabezas se entrenan desde cero (solo el backbone admite transfer learning).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fundidora import FundidoraPCExtendida, bloque_conv


class CabezaClasificacion(nn.Module):
    def __init__(self, c_in: int, n_salidas: int = 6, p_drop: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * c_in, 256), nn.ReLU(inplace=True),
                                 nn.Dropout(p_drop), nn.Linear(256, n_salidas))

    def forward(self, f):
        # avg: "cuanto hueso de cada tipo hay"; max: "hay al menos un patron fuerte en algun lado"
        v = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return self.mlp(v)


class CabezaDeteccion(nn.Module):
    def __init__(self, c_in: int, n_clases: int = 3, c_oculto: int = 256, prior: float = 0.01):
        super().__init__()
        self.tronco = nn.Sequential(nn.Conv2d(c_in, c_oculto, 3, padding=1), nn.ReLU(inplace=True),
                                    nn.Conv2d(c_oculto, c_oculto, 3, padding=1), nn.ReLU(inplace=True))
        self.cls = nn.Conv2d(c_oculto, n_clases, 1)
        self.box = nn.Conv2d(c_oculto, 4, 1)
        # sesgo inicial: prob. de objeto = 0.01 en todas las celdas (evita que la perdida
        # de clasificacion explote al inicio con 256 celdas casi todas negativas — RetinaNet)
        nn.init.constant_(self.cls.bias, -math.log((1 - prior) / prior))

    def forward(self, f):
        h = self.tronco(f)
        # ltrb en unidades de celda; softplus garantiza distancias positivas
        return self.cls(h), F.softplus(self.box(h))


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.conv = bloque_conv(c_in + c_skip, c_out, n_convs=2)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DecoderUNet(nn.Module):
    def __init__(self, canales_enc, canales_dec):
        super().__init__()
        c = canales_enc[-1]
        self.ups = nn.ModuleList()
        for skip, out in zip(reversed(canales_enc[:-1]), canales_dec):
            self.ups.append(UpBlock(c, skip, out))
            c = out
        self.c_out = c

    def forward(self, feats):
        x = feats[-1]
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            x = up(x, skip)
        return x


class CabezaSegmentacion(nn.Module):
    def __init__(self, c_in: int, n_sem=4, n_rol=3):
        super().__init__()
        self.sem = nn.Conv2d(c_in, n_sem, 1)
        self.rol = nn.Conv2d(c_in, n_rol, 1)
        self.borde = nn.Conv2d(c_in, 1, 1)

    def forward(self, d):
        return self.sem(d), self.rol(d), self.borde(d)


class CabezaDistancia(nn.Module):
    def __init__(self, c_in: int, c_oculto: int = 32):
        super().__init__()
        self.red = nn.Sequential(nn.Conv2d(c_in, c_oculto, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(c_oculto, 1, 1))

    def forward(self, d):
        return F.softplus(self.red(d))  # log(1 + mm) >= 0


class PengwinNet(nn.Module):
    TODAS = ("cls", "det", "seg", "dist")

    def __init__(self, cfg_modelo: dict, in_ch: int = 3):
        super().__init__()
        m = cfg_modelo
        self.backbone = FundidoraPCExtendida(in_ch, m["canales"], m["convs_por_bloque"],
                                             m["cbam_en_bloques"], m["cbam_reduccion"])
        c5 = m["canales"][-1]
        dec = m["canales_decoder"][: len(m["canales"]) - 1]
        self.cabeza_cls = CabezaClasificacion(c5)
        self.cabeza_det = CabezaDeteccion(c5)
        self.decoder = DecoderUNet(m["canales"], dec)
        self.cabeza_seg = CabezaSegmentacion(self.decoder.c_out)
        self.cabeza_dist = CabezaDistancia(self.decoder.c_out)

    def forward(self, x, cabezas=TODAS) -> dict:
        feats = self.backbone.forward_features(x)
        f5 = feats[-1]
        out = {}
        if "cls" in cabezas:
            out["cls"] = self.cabeza_cls(f5)
        if "det" in cabezas:
            out["det_cls"], out["det_box"] = self.cabeza_det(f5)
        if "seg" in cabezas or "dist" in cabezas:   # el decoder solo corre si se necesita
            d = self.decoder(feats)
            if "seg" in cabezas:
                out["sem"], out["rol"], out["borde"] = self.cabeza_seg(d)
            if "dist" in cabezas:
                out["dist"] = self.cabeza_dist(d)
        return out

    def congelar_backbone(self, congelar: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not congelar
