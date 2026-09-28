"""Semana 8: construye la cache de cortes preprocesados y los splits fijos por paciente.

    python scripts/preparar_datos.py --config configs/default.yaml

Se corre UNA vez (tarda: lee los ~7.6 GB). Es reanudable: salta los casos ya en cache.
Genera:
    data/cache/<id>/{img,lbl,dist}.npy + meta.json
    data/cache/resumen.json       (spacing, forma, etiquetas por caso -> insumo del EDA)
    data/splits.json              (train/val/test por paciente + SHA-256)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io import listar_casos  # noqa: E402
from src.data.preprocesamiento import preparar_caso  # noqa: E402
from src.data.splits import crear_splits, guardar_splits  # noqa: E402
from src.utils import cargar_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--rehacer", action="store_true", help="ignora la cache existente")
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    R, D = cfg["rutas"], cfg["datos"]

    casos = listar_casos(R["imagenes"], R["etiquetas"])
    print(f"{len(casos)} casos encontrados en {R['imagenes']}")
    if not casos:
        print("No hay casos. Revisa rutas.imagenes / rutas.etiquetas en la config.")
        return
    resumenes = []
    for i, c in enumerate(casos, 1):
        meta = Path(R["cache"]) / c["id"] / "meta.json"
        if meta.exists() and not args.rehacer:
            resumenes.append(json.load(open(meta, encoding="utf-8")))
            continue
        t0 = time.time()
        try:
            r = preparar_caso(c["imagen"], c["etiqueta"], R["cache"], D["resolucion"], D["ventana_hu"])
        except Exception as e:  # un caso corrupto no debe tumbar todo el proceso
            print(f"[error] {c['id']}: {e}")
            continue
        resumenes.append(r)
        print(f"[{i}/{len(casos)}] {c['id']}  forma {r['forma_zyx']}  spacing {[round(s, 3) for s in r['spacing_zyx']]}"
              f"  cortes hueso {r.get('cortes_con_hueso')}  {time.time() - t0:.1f}s")

    with open(Path(R["cache"]) / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumenes, f, indent=1)
    if Path(R["splits"]).exists() and not args.rehacer:
        print(f"Splits ya existen en {R['splits']} (no se tocan: son fijos). Usa --rehacer para regenerarlos.")
        return
    sp = crear_splits(resumenes, tuple(D["split"]), cfg["seed"])
    guardar_splits(sp, R["splits"])
    print({k: len(v) for k, v in sp["splits"].items()}, "sha256", sp["sha256"][:16])


if __name__ == "__main__":
    main()
