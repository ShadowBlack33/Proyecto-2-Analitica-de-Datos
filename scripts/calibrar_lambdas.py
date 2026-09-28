"""Calibracion de los lambdas de la perdida compuesta (requisito 4.1: justificados, no copiados).

Idea (la misma medicion de S6, donde sin ponderar L_cls dominaba ~45x a L_box):
en el modelo recien inicializado, para cada termino L_i se mide
    - su magnitud promedio
    - la norma del gradiente que produce sobre la ultima capa COMPARTIDA del backbone
Un termino con gradiente 20x mayor decide solo hacia donde se mueve el backbone.
Se propone lambda_i proporcional a 1 / ||grad_i|| (normalizado para que lambda_sem = 1),
y luego se redondea y se ajusta con criterio (p.ej. dar mas peso a clasificacion, que el
profesor senalo como la parte mas importante). La tabla que imprime va al informe.

    python scripts/calibrar_lambdas.py --lotes 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PengwinCortes  # noqa: E402
from src.data.splits import cargar_splits  # noqa: E402
from src.entrenamiento import mover  # noqa: E402
from src.losses.perdidas import PerdidaPengwin  # noqa: E402
from src.models.modelo import PengwinNet  # noqa: E402
from src.utils import cargar_config, dispositivo, fijar_semilla  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--lotes", type=int, default=20)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    fijar_semilla(cfg["seed"])
    dev = dispositivo(args.device)
    splits = cargar_splits(cfg["rutas"]["splits"])
    ds = PengwinCortes(cfg["rutas"]["cache"], splits["train"], cfg, entrenamiento=True)
    dl = DataLoader(ds, cfg["entrenamiento"]["batch_size"], shuffle=True, num_workers=0)

    modelo = PengwinNet(cfg["modelo"], in_ch=2 * cfg["datos"]["contexto"] + 1).to(dev).train()
    perdida = PerdidaPengwin(cfg)
    # quitar los lambdas actuales para medir cada termino "crudo"
    for k in list(perdida.c):
        if k.startswith("lambda_"):
            perdida.c[k] = 1.0
    compartido = [p for p in modelo.backbone.bloques[-1].parameters()]
    mags, grads = {}, {}
    for i, batch in enumerate(dl):
        if i >= args.lotes:
            break
        batch = mover(batch, dev)
        out = modelo(batch["img"])
        _, comp = perdida(out, batch, PengwinNet.TODAS)
        # recalcular cada termino por separado para su gradiente
        for k in comp:
            modelo.zero_grad()
            out = modelo(batch["img"])
            perdida_k = PerdidaPengwin(cfg)
            for kk in list(perdida_k.c):
                if kk.startswith("lambda_"):
                    perdida_k.c[kk] = 0.0
            nombre_lambda = {"det_obj": "lambda_det_obj", "det_box": "lambda_det_box"}.get(k, f"lambda_{k}")
            perdida_k.c[nombre_lambda] = 1.0
            total, c = perdida_k(out, batch, PengwinNet.TODAS)
            total.backward()
            g = torch.sqrt(sum((p.grad ** 2).sum() for p in compartido if p.grad is not None))
            mags.setdefault(k, []).append(c[k])
            grads.setdefault(k, []).append(float(g))

    ref = np.mean(grads["sem"])
    print(f"\n{'termino':10s} {'magnitud':>10s} {'||grad||':>10s} {'grad/ref':>9s} {'lambda sugerido':>16s}")
    for k in mags:
        g = np.mean(grads[k])
        print(f"{k:10s} {np.mean(mags[k]):10.4f} {g:10.4f} {g / ref:9.2f} {ref / max(g, 1e-9):16.2f}")
    print("\nreferencia: lambda_sem = 1. Redondear y justificar en el informe; ajustar con validacion.")


if __name__ == "__main__":
    main()
