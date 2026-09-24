"""Inferencia sobre un CT y exportacion: mascara con etiquetas PENGWIN (.mha, abre en 3D Slicer),
tabla de fragmentos con distancias (JSON) y latencia. Util para el video de respaldo del pitch.

    python scripts/inferir.py --ckpt checkpoints/exp/mejor_conjunto.pth --ct data/raw/images/001.mha
    python scripts/inferir.py --ckpt ... --ct ... --device cpu        # latencia en CPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io import guardar_volumen, id_caso, leer_volumen  # noqa: E402
from src.inferencia import cargar_modelo, inferir_volumen  # noqa: E402
from src.utils import cargar_config, dispositivo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    dev = dispositivo(args.device)
    modelo = cargar_modelo(args.ckpt, cfg, dev)
    hu, sp, meta = leer_volumen(args.ct)
    res = inferir_volumen(modelo, hu, sp, cfg, dev, amp=cfg["entrenamiento"]["amp"])

    out = Path(args.salida or Path(cfg["rutas"]["resultados"]) / "inferencias")
    out.mkdir(parents=True, exist_ok=True)
    cid = id_caso(args.ct)
    guardar_volumen(res["etiquetas"], sp, out / f"{cid}_pred.mha", meta)
    with open(out / f"{cid}_fragmentos.json", "w", encoding="utf-8") as f:
        json.dump({"tabla": res["tabla"], "latencia": res["latencia"]}, f, indent=1, default=float)
    print(f"{len(res['tabla'])} fragmentos | {res['latencia']['ms_por_corte']:.1f} ms/corte en {dev.type}")
    for fila in res["tabla"]:
        extra = "principal" if fila["es_principal"] else f"{fila['distancia_mm']} mm (cabeza: {fila['distancia_cabeza_mm']} mm)"
        print(f"  {fila['nombre']:6s} {fila['volumen_ml']:8.2f} ml  {extra}")
    print(f"resultados en {out}")


if __name__ == "__main__":
    main()
