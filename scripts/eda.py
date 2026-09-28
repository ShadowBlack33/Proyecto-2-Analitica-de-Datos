"""EDA semana 8 — caracteriza el dataset ANTES de decidir nada.

    python scripts/eda.py --config configs/default.yaml            # todos los casos
    python scripts/eda.py --max_casos 10                             # prueba rapida

Salida en resultados/eda/:
    eda_casos.csv        una fila por caso: forma, spacing, n fragmentos por region, calidad
    eda_fragmentos.csv   una fila por fragmento: region, volumen, distancia GT al principal (mm)
    *.png                histogramas para el informe
Preguntas que responde (y que justifican decisiones):
  * spacing en z (espesor de corte): cuanto varia entre casos -> por que medir en mm con el header
  * fragmentos por region: cuantos casos tienen 0, 1-3, 4+ secundarios -> estratificacion del split
  * distancias GT: rango real (el profesor dijo 2 mm a 25 cm) -> justifica log(1+d) / Huber
  * calidad (nitidez, ruido): terciles -> subconjunto de evaluacion por calidad
  * HU del hueso: percentiles -> ventana HU y umbral del MIP
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.calidad import perfil_calidad  # noqa: E402
from src.data.etiquetas import validar_etiquetas  # noqa: E402
from src.data.io import leer_volumen, listar_casos  # noqa: E402
from src.postprocess.distancia import tabla_fragmentos  # noqa: E402
from src.utils import cargar_config  # noqa: E402


def escribir_csv(filas, ruta):
    if not filas:
        return
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def graficas(casos, frags, dir_out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no instalado: se omiten las graficas")
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist([c["spacing_z_mm"] for c in casos], bins=20, color="#23716F")
    ax[0, 0].set_title("Espesor de corte (mm)")
    for i, r in enumerate(["SA", "LI", "RI"]):
        ax[0, 1].bar(np.arange(11) + 0.27 * i, np.bincount([c[f"n_frag_{r}"] for c in casos], minlength=11)[:11],
                     width=0.27, label=r)
    ax[0, 1].set_title("Fragmentos por region (casos)")
    ax[0, 1].legend()
    d = [f["distancia_mm"] for f in frags if f["distancia_mm"] is not None]
    if d:
        ax[1, 0].hist(d, bins=40, color="#A66A16")
        ax[1, 0].set_yscale("log")
        ax[1, 0].set_title(f"Distancia GT al principal (mm) — max {max(d):.1f}")
    ax[1, 1].hist([c["nitidez_lap"] for c in casos], bins=20, color="#5B584C")
    ax[1, 1].set_title("Nitidez (varianza del Laplaciano)")
    fig.tight_layout()
    fig.savefig(dir_out / "eda_resumen.png", dpi=130)
    print(f"grafica: {dir_out / 'eda_resumen.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max_casos", type=int, default=None)
    args = ap.parse_args()
    cfg = cargar_config(args.config)
    R = cfg["rutas"]
    dir_out = Path(R["resultados"]) / "eda"
    dir_out.mkdir(parents=True, exist_ok=True)

    casos, frags, hu_hueso = [], [], []
    lista = listar_casos(R["imagenes"], R["etiquetas"])[: args.max_casos]
    for i, c in enumerate(lista, 1):
        hu, sp, _ = leer_volumen(c["imagen"])
        fila = {"id": c["id"], "Z": hu.shape[0], "H": hu.shape[1], "W": hu.shape[2],
                "spacing_z_mm": sp[0], "spacing_y_mm": sp[1], "spacing_x_mm": sp[2],
                "hu_min": int(hu.min()), "hu_max": int(hu.max())}
        fila.update(perfil_calidad(hu, sp, tuple(cfg["datos"]["ventana_hu"])))
        if c["etiqueta"] is not None:
            gt, _, _ = leer_volumen(c["etiqueta"])
            v = validar_etiquetas(gt)
            if v["fuera_de_rango"]:
                print(f"[AVISO] {c['id']}: etiquetas fuera de 0..30: {v['fuera_de_rango']}")
            for r, nom in zip((1, 2, 3), ("SA", "LI", "RI")):
                fila[f"n_frag_{nom}"] = sum(1 for e in v["valores"] if e > 0 and (e - 1) // 10 + 1 == r)
            m = gt > 0
            hu_hueso.append(np.percentile(hu[m], [1, 5, 50, 95, 99]) if m.any() else np.full(5, np.nan))
            for f in tabla_fragmentos(gt, sp):
                frags.append({"caso": c["id"], "fragmento": f["nombre"], "principal": f["es_principal"],
                              "volumen_ml": f["volumen_ml"], "distancia_mm": f["distancia_mm"]})
        casos.append(fila)
        print(f"[{i}/{len(lista)}] {c['id']}  {fila['Z']}x{fila['H']}x{fila['W']}  sz {sp[0]:.2f}mm  "
              f"frag SA/LI/RI {fila.get('n_frag_SA')}/{fila.get('n_frag_LI')}/{fila.get('n_frag_RI')}")

    escribir_csv(casos, dir_out / "eda_casos.csv")
    escribir_csv(frags, dir_out / "eda_fragmentos.csv")
    if hu_hueso:
        p = np.nanmedian(np.array(hu_hueso), 0)
        print(f"\nHU dentro del hueso (mediana entre casos) p1/p5/p50/p95/p99: {np.round(p).tolist()}")
    d = [f["distancia_mm"] for f in frags if f["distancia_mm"] is not None]
    if d:
        print(f"Distancias GT: n={len(d)}  min {min(d):.1f}  mediana {np.median(d):.1f}  p95 "
              f"{np.percentile(d, 95):.1f}  max {max(d):.1f} mm")
    graficas(casos, frags, dir_out)
    print(f"CSV en {dir_out}")


if __name__ == "__main__":
    main()
