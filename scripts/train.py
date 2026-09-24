"""Entrenamiento por curriculum (configs/default.yaml -> entrenamiento.etapas).

Uso:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --reanudar checkpoints/ultimo.pth
    python scripts/train.py --config configs/default.yaml --etapas deteccion   # solo una etapa

Etapas por defecto: deteccion -> segmentacion -> distancia (backbone congelado) -> conjunto.
No se entrenan las 4 cabezas desde el dia uno: si algo falla, se sabe cual cabeza fue.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PengwinCortes  # noqa: E402
from src.data.splits import cargar_splits  # noqa: E402
from src.entrenamiento import RegistroCSV, entrenar_epoca, guardar_checkpoint, validar  # noqa: E402
from src.losses.perdidas import PerdidaPengwin  # noqa: E402
from src.models.fundidora import cargar_pesos_parciales  # noqa: E402
from src.models.modelo import PengwinNet  # noqa: E402
from src.utils import cargar_config, contar_parametros, dispositivo, fijar_semilla  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--reanudar", default=None)
    ap.add_argument("--etapas", nargs="*", default=None, help="nombres de etapas a correr")
    ap.add_argument("--nombre", default="exp", help="nombre del experimento (ablation: sin_cbam, sin_tl...)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = cargar_config(args.config)
    fijar_semilla(cfg["seed"])
    dev = dispositivo(args.device)
    R, E = cfg["rutas"], cfg["entrenamiento"]
    fast = bool(E["fast_dev_run"])
    splits = cargar_splits(R["splits"])

    ds_tr = PengwinCortes(R["cache"], splits["train"], cfg, entrenamiento=True, seed=cfg["seed"])
    ds_va = PengwinCortes(R["cache"], splits["val"], cfg, entrenamiento=False)
    g = torch.Generator().manual_seed(cfg["seed"])
    nw = cfg["datos"]["num_workers"]
    dl_tr = DataLoader(ds_tr, E["batch_size"], shuffle=True, num_workers=nw, drop_last=True, generator=g,
                       pin_memory=dev.type == "cuda")
    dl_va = DataLoader(ds_va, E["batch_size"], shuffle=False, num_workers=nw)
    print(f"[datos] train {len(ds_tr)} cortes de {len(splits['train'])} casos | val {len(ds_va)} cortes")

    modelo = PengwinNet(cfg["modelo"], in_ch=2 * cfg["datos"]["contexto"] + 1).to(dev)
    if cfg["modelo"].get("pesos_backbone"):
        cargar_pesos_parciales(modelo.backbone, cfg["modelo"]["pesos_backbone"])
    print(f"[modelo] {contar_parametros(modelo):,} parametros | dispositivo {dev}")

    perdida = PerdidaPengwin(cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(E["amp"]) and dev.type == "cuda")
    lote_fijo = next(iter(dl_va))["img"][:4].to(dev)
    dir_ck = Path(R["checkpoints"]) / args.nombre
    log = RegistroCSV(Path(R["resultados"]) / args.nombre / "log_entrenamiento.csv")

    inicio_etapa, inicio_epoca = 0, 0
    ck = None
    if args.reanudar:
        ck = torch.load(args.reanudar, map_location="cpu", weights_only=False)
        modelo.load_state_dict(ck["modelo"])
        nombres = [e["nombre"] for e in E["etapas"]]
        inicio_etapa, inicio_epoca = nombres.index(ck["etapa"]), ck["epoca"] + 1
        print(f"[reanudar] etapa {ck['etapa']} desde epoca {inicio_epoca}")

    for ie, etapa in enumerate(E["etapas"]):
        if ie < inicio_etapa or (args.etapas and etapa["nombre"] not in args.etapas):
            continue
        cabezas = tuple(etapa["cabezas"])
        modelo.congelar_backbone(bool(etapa.get("congelar_backbone", False)))
        params = [p for p in modelo.parameters() if p.requires_grad]
        lr = E["lr"] * etapa.get("lr_factor", 1.0)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=E["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, etapa["epocas"]))
        if ck is not None and ie == inicio_etapa and "optimizador" in ck:
            try:
                opt.load_state_dict(ck["optimizador"])
                scaler.load_state_dict(ck["scaler"])
            except ValueError:
                pass
        mejor = -float("inf")
        print(f"\n=== etapa {etapa['nombre']} | cabezas {cabezas} | backbone "
              f"{'congelado' if etapa.get('congelar_backbone') else 'entrenable'} | lr {lr}")
        for ep in range(inicio_epoca if ie == inicio_etapa else 0, etapa["epocas"]):
            t0 = time.time()
            lr_actual = opt.param_groups[0]["lr"]
            tr = entrenar_epoca(modelo, dl_tr, perdida, opt, scaler, dev, cabezas, E, fast)
            va = validar(modelo, dl_va, perdida, dev, cabezas, cfg, fast)
            sched.step()
            fila = {"etapa": etapa["nombre"], "epoca": ep, "lr": lr_actual,
                    "seg_epoca": round(time.time() - t0, 1), **tr, **va}
            log.escribir(fila)
            print(f"[{etapa['nombre']} {ep:02d}] loss {tr['total']:.4f} | val {va['val_total']:.4f} "
                  f"| score {va['val_score']:.4f} | {fila['seg_epoca']}s")
            guardar_checkpoint(dir_ck / "ultimo.pth", modelo, opt, scaler, etapa["nombre"], ep, cfg, va)
            if va["val_score"] > mejor:
                mejor = va["val_score"]
                guardar_checkpoint(dir_ck / f"mejor_{etapa['nombre']}.pth", modelo, opt, scaler,
                                   etapa["nombre"], ep, cfg, va, lote_fijo)
                print(f"   -> nuevo mejor ({mejor:.4f})")
            if fast:
                break
        # la siguiente etapa arranca desde el MEJOR de esta, no desde el ultimo
        mejor_ck = torch.load(dir_ck / f"mejor_{etapa['nombre']}.pth", map_location=dev, weights_only=False)
        modelo.load_state_dict(mejor_ck["modelo"])
        ck = None
    modelo.congelar_backbone(False)
    ultima = [e["nombre"] for e in E["etapas"] if not args.etapas or e["nombre"] in args.etapas][-1]
    print(f"\nListo. Modelo final: {dir_ck / f'mejor_{ultima}.pth'}")


if __name__ == "__main__":
    main()
