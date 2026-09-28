"""Mide la VRAM pico real (forward + backward con AMP) antes de comprometerse a un batch.

Correrlo en el PC MAS LIMITADO del equipo (RTX 3050, 4 GB) antes de la semana 9:
    python scripts/perfil_memoria.py
    python scripts/perfil_memoria.py --resoluciones 224 256 --batches 4 8 12 16

Si un batch no cabe, se usa uno menor y se sube `entrenamiento.acumulacion` para mantener
el batch efectivo (batch_size x acumulacion).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.modelo import PengwinNet  # noqa: E402
from src.utils import cargar_config, contar_parametros  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resoluciones", nargs="+", type=int, default=[224, 256])
    ap.add_argument("--batches", nargs="+", type=int, default=[4, 8, 12, 16])
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    modelo = PengwinNet(cfg["modelo"], in_ch=2 * cfg["datos"]["contexto"] + 1)
    print(f"parametros: {contar_parametros(modelo):,}")
    if not torch.cuda.is_available():
        print("Sin GPU: este script mide VRAM, correrlo en el PC con la RTX.")
        return
    dev = torch.device("cuda")
    modelo = modelo.to(dev).train()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)\n")
    print(f"{'res':>5} {'batch':>6} {'pico GB':>9}  estado")
    opt = torch.optim.AdamW(modelo.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda")
    for res in args.resoluciones:
        for b in args.batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                x = torch.randn(b, 3, res, res, device=dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = modelo(x)
                    loss = sum(v.float().mean() for v in out.values())
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                pico = torch.cuda.max_memory_allocated() / 1024 ** 3
                estado = "OK" if pico < 0.85 * total_gb else "justo (riesgo de OOM)"
                print(f"{res:5d} {b:6d} {pico:9.2f}  {estado}")
            except torch.cuda.OutOfMemoryError:
                print(f"{res:5d} {b:6d} {'—':>9}  NO CABE")
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
