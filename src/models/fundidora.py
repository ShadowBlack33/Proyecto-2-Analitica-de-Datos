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


def _unidades_conv_bn(sd: dict) -> list[dict]:
    """Agrupa un state_dict en unidades conv (+bias opcional) -> BatchNorm, en orden."""
    claves = list(sd)
    unidades = []
    for i, k in enumerate(claves):
        if not (k.endswith("weight") and sd[k].ndim == 4):
            continue
        pref = k[: -len("weight")]
        bn = next((c[: -len("running_mean")] for c in claves[i + 1:] if c.endswith("running_mean")), None)
        if bn is None:
            break
        unidades.append({"w": k, "b": pref + "bias" if pref + "bias" in sd else None, "bn": bn})
    return unidades


def cargar_pesos_parciales(modelo: nn.Module, ruta: str, verbose=True) -> int:
    """Transfer learning en el backbone (unico lugar donde el enunciado lo permite).

    Copia los pesos de una FundidoraPC entrenada antes en el curso (p. ej. el backbone del
    Taller 3) a los bloques equivalentes de la extendida. Empareja por capa y en orden
    (conv con conv, BatchNorm con BatchNorm), no por nombre, porque cada notebook nombro
    las capas distinto. Detalles:
    - Si viene de un detector completo, usa solo las capas cuyo nombre contiene "backbone".
    - Si la conv de origen tiene bias y la nuestra no (va seguida de BatchNorm), el bias se
      absorbe en la media del BatchNorm: BN(x + b) con media m == BN(x) con media m - b.
    - Si la primera conv se entreno con 1 canal, se replica a los canales de entrada.
    - Se detiene donde la arquitectura diverge (el 5.o bloque es nuevo y arranca de cero).
    Devuelve cuantos bloques conv+BN se copiaron (reportarlo en el model card).
    """
    fuente = torch.load(ruta, map_location="cpu", weights_only=False)
    for clave in ("modelo", "model_state_dict", "state_dict", "model"):
        if isinstance(fuente, dict) and isinstance(fuente.get(clave), (dict, nn.Module)):
            fuente = fuente[clave]
            break
    if isinstance(fuente, nn.Module):
        fuente = fuente.state_dict()
    fuente = {k: v for k, v in fuente.items() if torch.is_tensor(v)}
    del_backbone = {k: v for k, v in fuente.items() if "backbone" in k}
    if del_backbone:
        fuente = del_backbone

    destino = modelo.state_dict()
    u_src = _unidades_conv_bn(fuente)
    u_dst = _unidades_conv_bn({k: v for k, v in destino.items() if k.startswith("bloques")})
    copiados = 0
    for us, ud in zip(u_src, u_dst):
        ws, wd = fuente[us["w"]], destino[ud["w"]]
        if ws.shape[0] != wd.shape[0] or ws.shape[2:] != wd.shape[2:]:
            break
        if ws.shape[1] != wd.shape[1]:
            if copiados > 0:
                break
            ws = ws.mean(1, keepdim=True).repeat(1, wd.shape[1], 1, 1)
        destino[ud["w"]] = ws.clone()
        for t in ("weight", "bias", "running_var"):
            destino[ud["bn"] + t] = fuente[us["bn"] + t].clone()
        media = fuente[us["bn"] + "running_mean"].clone()
        if us["b"] is not None and ud["b"] is None:
            media = media - fuente[us["b"]]
        destino[ud["bn"] + "running_mean"] = media
        copiados += 1
    modelo.load_state_dict(destino)
    if verbose:
        print(f"[transfer] {copiados} de {len(u_dst)} bloques conv+BN copiados desde {ruta}")
    if copiados == 0:
        raise RuntimeError(f"No se pudo copiar ningun bloque desde {ruta}: revisar la arquitectura del checkpoint")
    return copiados
