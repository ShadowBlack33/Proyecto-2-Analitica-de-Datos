"""Genera CT sinteticos con la MISMA estructura que PENGWIN (.mha, etiquetas 0..30)
para probar todo el pipeline de punta a punta sin los 7.6 GB reales.

NO sirven para medir desempeno: solo para verificar que el codigo corre
(formas, spacing, etiquetas, cache, entrenamiento, inferencia, dashboard).

    python scripts/generar_datos_sinteticos.py --n 8 --salida data/sintetico
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.io import guardar_volumen  # noqa: E402


def elipsoide(forma, centro, radios):
    z, y, x = np.ogrid[:forma[0], :forma[1], :forma[2]]
    return (((z - centro[0]) / radios[0]) ** 2 + ((y - centro[1]) / radios[1]) ** 2
            + ((x - centro[2]) / radios[2]) ** 2) <= 1.0


def caso(rng, forma=(40, 128, 160)):
    Z, H, W = forma
    hu = np.full(forma, -1000.0, np.float32)
    lbl = np.zeros(forma, np.uint8)
    cuerpo = elipsoide(forma, (Z / 2, H / 2, W / 2), (Z, H * 0.42, W * 0.45))
    hu[cuerpo] = 40
    # sacro al centro, coxales a los lados (x grande = izquierda del paciente en LPS)
    sa = elipsoide(forma, (Z / 2, H * 0.55, W / 2), (Z * 0.35, 12, 10))
    li = elipsoide(forma, (Z / 2, H * 0.5, W * 0.72), (Z * 0.4, 22, 14))
    ri = elipsoide(forma, (Z / 2, H * 0.5, W * 0.28), (Z * 0.4, 22, 14))
    lbl[sa], lbl[li], lbl[ri] = 1, 11, 21
    # fragmentos: uno desplazado (con separacion) y otro en contacto
    if rng.random() < 0.8:
        dz = int(rng.integers(3, 8))
        frag = elipsoide(forma, (Z / 2 + dz, H * 0.35, W * 0.78), (4, 6, 5))
        lbl[frag] = 12
    if rng.random() < 0.8:
        frag = elipsoide(forma, (Z / 2, H * 0.5, W * 0.28 - 12), (6, 8, 5))
        lbl[frag & (lbl == 21)] = 22
        frag2 = elipsoide(forma, (Z / 2 - 8, H * 0.30, W * 0.22), (3, 5, 4))
        lbl[frag2] = 23
    hu[lbl > 0] = 700 + rng.normal(0, 60, int((lbl > 0).sum()))
    hu += rng.normal(0, float(rng.uniform(10, 60)), forma)   # calidad variable entre casos
    return hu.astype(np.int16), lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--salida", default="data/sintetico")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    di, dl = Path(args.salida) / "images", Path(args.salida) / "labels"
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.n + 1):
        hu, lbl = caso(rng)
        spacing = (2.5, 0.8, 0.8)
        guardar_volumen(hu, spacing, di / f"{i:03d}.mha")
        guardar_volumen(lbl, spacing, dl / f"{i:03d}.mha")
    print(f"{args.n} casos sinteticos en {args.salida}")


if __name__ == "__main__":
    main()
