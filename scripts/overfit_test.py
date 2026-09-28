"""Semana 9: overfit intencional sobre un lote pequeno como prueba de correctitud.

Si el modelo NO logra memorizar 8 cortes, hay un bug (targets mal construidos, perdida
mal conectada, gradiente que no llega) y no tiene sentido lanzar un entrenamiento largo.
Criterio: la perdida total cae > 90% y las cajas predichas sobre esos mismos cortes
tienen IoU alto con las reales.

    python scripts/overfit_test.py --cabezas cls det            # semana 9
    python scripts/overfit_test.py --cabezas cls det seg dist   # pipeline completo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PengwinCortes  # noqa: E402
from src.data.splits import cargar_splits  # noqa: E402
from src.entrenamiento import mover  # noqa: E402
from src.eval.metricas_det import resumen_deteccion  # noqa: E402
from src.losses.perdidas import PerdidaPengwin  # noqa: E402
from src.models.modelo import PengwinNet  # noqa: E402
from src.postprocess.nms import detecciones  # noqa: E402
from src.utils import cargar_config, dispositivo, fijar_semilla  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cabezas", nargs="+", default=["cls", "det"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--pasos", type=int, default=300)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = cargar_config(args.config)
    fijar_semilla(cfg["seed"])
    dev = dispositivo(args.device)
    splits = cargar_splits(cfg["rutas"]["splits"])
    ds = PengwinCortes(cfg["rutas"]["cache"], splits["train"][:2], cfg, entrenamiento=False)
    # cortes con hueso, repartidos a lo largo del volumen
    con_hueso = [i for i in range(len(ds)) if ds[i]["hay_caja"].any()]
    idx = con_hueso[:: max(1, len(con_hueso) // args.n)][: args.n]
    batch = torch.utils.data.default_collate([ds[i] for i in idx])
    batch = mover(batch, dev)

    modelo = PengwinNet(cfg["modelo"], in_ch=2 * cfg["datos"]["contexto"] + 1).to(dev)
    perdida = PerdidaPengwin(cfg)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    cabezas = tuple(args.cabezas)
    usar_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)
    inicial = None
    for paso in range(args.pasos):
        modelo.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp):
            out = modelo(batch["img"], cabezas)
        total, comp = perdida(out, batch, cabezas)
        opt.zero_grad()
        scaler.scale(total).backward()
        scaler.step(opt)
        scaler.update()
        if inicial is None:
            inicial = float(total.detach())
        if paso % 25 == 0 or paso == args.pasos - 1:
            print(f"paso {paso:4d}  total {total.item():.4f}  " + "  ".join(f"{k} {v:.3f}" for k, v in comp.items()))

    caida = 1 - float(total.detach()) / inicial
    print(f"\nperdida inicial {inicial:.4f} -> final {float(total.detach()):.4f}  (caida {100 * caida:.1f}%)")
    if "det" in cabezas:
        modelo.eval()
        with torch.no_grad():
            o = modelo(batch["img"], cabezas)
        dc = cfg["deteccion"]
        preds = detecciones(o["det_cls"], o["det_box"], dc["stride"], 0.05, dc["umbral_nms"], 1,
                            cfg["datos"]["resolucion"])
        gts = [[{"clase": r, "caja": batch["cajas"][b, r].tolist()} for r in range(3) if bool(batch["hay_caja"][b, r])]
               for b in range(len(idx))]
        rd = resumen_deteccion(preds, gts)
        print(f"IoU promedio de cajas sobre el lote memorizado: {rd['iou_promedio']:.3f}  mAP@0.5 {rd['mAP@0.50']:.3f}")
    print("OK: el modelo puede memorizar" if caida > 0.9 else "REVISAR: la perdida no cae lo suficiente")


if __name__ == "__main__":
    main()
