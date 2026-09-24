"""Baseline zero-shot con SAM (seccion 4.2). SOLO comparacion: nunca entra al pipeline
ni genera etiquetas.

Protocolo: sobre el split de test, SAM recibe como prompt la caja PREDICHA por nuestro
modelo en cada corte y devuelve una mascara. Como una caja por region no puede separar
fragmentos, la comparacion justa es a nivel de REGION (SA/LI/RI): Dice e IoU 3D de SAM vs
Dice e IoU 3D de nuestro modelo, ambos contra el ground truth. Opcionalmente tambien con
la caja del GT (--caja_gt) como techo de SAM.

Requisitos (no van en requirements.txt porque pesan):
    pip install git+https://github.com/facebookresearch/segment-anything.git
    descargar sam_vit_b_01ec64.pth (el mas liviano: cabe en 4 GB) a checkpoints/

    python scripts/baseline_sam.py --ckpt checkpoints/exp/mejor_conjunto.pth --sam checkpoints/sam_vit_b_01ec64.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data.etiquetas import region_de_etiqueta  # noqa: E402
from src.data.io import leer_volumen, listar_casos  # noqa: E402
from src.data.preprocesamiento import ventanear  # noqa: E402
from src.data.splits import cargar_splits  # noqa: E402
from src.eval.metricas_seg_cls import dice_iou  # noqa: E402
from src.inferencia import cargar_modelo, inferir_volumen  # noqa: E402
from src.utils import cargar_config, dispositivo, fijar_semilla  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sam", required=True, help="ruta a sam_vit_b_01ec64.pth")
    ap.add_argument("--tipo", default="vit_b")
    ap.add_argument("--cada", type=int, default=2, help="evaluar 1 de cada N cortes (SAM es lento)")
    ap.add_argument("--caja_gt", action="store_true", help="usar la caja del GT como prompt (techo de SAM)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from segment_anything import SamPredictor, sam_model_registry

    cfg = cargar_config(args.config)
    fijar_semilla(cfg["seed"])
    dev = dispositivo(args.device)
    modelo = cargar_modelo(args.ckpt, cfg, dev)
    sam = sam_model_registry[args.tipo](checkpoint=args.sam).to(dev).eval()
    pred_sam = SamPredictor(sam)

    ids = set(cargar_splits(cfg["rutas"]["splits"])["test"])
    casos = [c for c in listar_casos(cfg["rutas"]["imagenes"], cfg["rutas"]["etiquetas"]) if c["id"] in ids]
    filas = []
    for c in casos:
        hu, sp, _ = leer_volumen(c["imagen"])
        gt, _, _ = leer_volumen(c["etiqueta"])
        res = inferir_volumen(modelo, hu, sp, cfg, dev)
        reg_gt, reg_prop = region_de_etiqueta(gt), region_de_etiqueta(res["etiquetas"])
        reg_sam = np.zeros_like(reg_gt)
        zs = range(0, hu.shape[0], args.cada)
        for z in zs:
            if args.caja_gt:
                cajas = []
                for r in (1, 2, 3):
                    m = reg_gt[z] == r
                    if m.any():
                        ys, xs = np.where(m)
                        cajas.append({"clase": r - 1, "caja": [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]})
            else:
                cajas = res["detecciones"][z]
            if not cajas:
                continue
            rgb = (np.repeat(ventanear(hu[z], cfg["datos"]["ventana_hu"])[..., None], 3, 2) * 255).astype(np.uint8)
            pred_sam.set_image(rgb)
            for d in cajas:
                with torch.no_grad():
                    m, _, _ = pred_sam.predict(box=np.array(d["caja"], dtype=np.float32), multimask_output=False)
                reg_sam[z][m[0] & (reg_sam[z] == 0)] = d["clase"] + 1
        sel = np.zeros(hu.shape[0], bool)
        sel[list(zs)] = True
        for r, nom in zip((1, 2, 3), ("SA", "LI", "RI")):
            g = reg_gt[sel] == r
            if not g.any():
                continue
            ds, js = dice_iou(g, reg_sam[sel] == r)
            dp, jp = dice_iou(g, reg_prop[sel] == r)
            filas.append({"caso": c["id"], "region": nom, "dice_sam": ds, "iou_sam": js, "dice_propio": dp, "iou_propio": jp})
            print(f"{c['id']} {nom}: SAM Dice {ds:.3f} IoU {js:.3f} | propio Dice {dp:.3f} IoU {jp:.3f}")

    resumen = {k: float(np.mean([f[k] for f in filas])) for k in ("dice_sam", "iou_sam", "dice_propio", "iou_propio")}
    print("\nRESUMEN (nivel region):", {k: round(v, 3) for k, v in resumen.items()})
    out = Path(cfg["rutas"]["resultados"]) / Path(args.ckpt).parent.name
    out.mkdir(parents=True, exist_ok=True)
    nombre = "baseline_sam_cajagt.json" if args.caja_gt else "baseline_sam.json"
    with open(out / nombre, "w", encoding="utf-8") as f:
        json.dump({"resumen": resumen, "detalle": filas}, f, indent=1)


if __name__ == "__main__":
    main()
