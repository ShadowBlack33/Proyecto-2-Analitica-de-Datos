"""Evaluacion final sobre el split de TEST (semana 10), a resolucion nativa y en 3D.

    python scripts/evaluar.py --ckpt checkpoints/exp/mejor_conjunto.pth
    python scripts/evaluar.py --ckpt ... --degradacion 0 1 2 3      # curva de robustez
    python scripts/evaluar.py --ckpt ... --device cpu               # latencia en CPU

Reporta (resultados/<nombre>/):
  * clasificacion: F1 y AUC (presencia SA/LI/RI por corte) + matriz de confusion por pixel
  * deteccion: IoU promedio, mAP@0.50, mAP@[0.50:0.95]
  * segmentacion de fragmento: Dice, IoU, HD95, ASSD (mm)
  * distancia: prediccion geometrica y de la cabeza vs ground truth (error en mm)
  * metricas por tercil de calidad de imagen y por nivel de degradacion sintetica
  * latencia por corte en el dispositivo usado
  * los 5 peores casos (insumo de la galeria del informe)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _media_nan(v):
    """nanmean sin avisos: NaN si no hay valores (p. ej. ningun fragmento emparejado)."""
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data import aumentos as A  # noqa: E402
from src.data.calidad import perfil_calidad, terciles  # noqa: E402
from src.data.etiquetas import region_de_etiqueta  # noqa: E402
from src.data.io import leer_volumen, listar_casos  # noqa: E402
from src.data.preprocesamiento import a_cuadrado, info_padding, redimensionar, ventanear  # noqa: E402
from src.data.splits import cargar_splits  # noqa: E402
from src.eval.metricas_det import resumen_deteccion  # noqa: E402
from src.eval.metricas_seg_cls import matriz_confusion_pixel, metricas_clasificacion, metricas_fragmentos  # noqa: E402
from src.inferencia import cargar_modelo, inferir_volumen  # noqa: E402
from src.postprocess.distancia import comparar_con_gt, tabla_fragmentos  # noqa: E402
from src.utils import cargar_config, dispositivo, fijar_semilla  # noqa: E402


def cajas_gt(etq: np.ndarray) -> list[list[dict]]:
    reg = region_de_etiqueta(etq)
    out = []
    for z in range(etq.shape[0]):
        cz = []
        for r in (1, 2, 3):
            m = reg[z] == r
            if m.any():
                ys, xs = np.where(m)
                cz.append({"clase": r - 1, "caja": [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]})
        out.append(cz)
    return out


def entrada_degradada(hu, cfg, nivel, semilla):
    tam = cfg["datos"]["resolucion"]
    info = info_padding(hu.shape[1], hu.shape[2])
    img = redimensionar(a_cuadrado(ventanear(hu, cfg["datos"]["ventana_hu"]), info, 0.0), tam, "bilinear")
    if nivel:
        torch.manual_seed(semilla)
        t = torch.from_numpy(img.astype(np.float32))
        img = torch.stack([A.degradar(t[i:i + 1], **A.NIVELES_DEGRADACION[nivel])[0] for i in range(t.shape[0])]).numpy()
    return img.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--degradacion", nargs="*", type=int, default=[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--nombre", default=None)
    ap.add_argument("--sin_superficie", action="store_true", help="omite HD95/ASSD (mas rapido)")
    args = ap.parse_args()

    cfg = cargar_config(args.config)
    fijar_semilla(cfg["seed"])
    dev = dispositivo(args.device)
    R = cfg["rutas"]
    modelo = cargar_modelo(args.ckpt, cfg, dev)
    ids = set(cargar_splits(R["splits"])[args.split])
    casos = [c for c in listar_casos(R["imagenes"], R["etiquetas"]) if c["id"] in ids]
    nombre = args.nombre or Path(args.ckpt).parent.name
    dir_out = Path(R["resultados"]) / nombre
    dir_out.mkdir(parents=True, exist_ok=True)

    por_nivel = {}
    calidad = {}
    for nivel in args.degradacion:
        y_cls, p_cls, preds, gts, filas_frag, filas_dist, por_caso, lat = [], [], [], [], [], [], [], []
        conf = np.zeros((4, 4), np.int64)
        for c in casos:
            hu, spacing, _ = leer_volumen(c["imagen"])
            gt, _, _ = leer_volumen(c["etiqueta"])
            if nivel == 0 and c["id"] not in calidad:
                calidad[c["id"]] = perfil_calidad(hu, spacing)
            x = entrada_degradada(hu, cfg, nivel, semilla=cfg["seed"])
            res = inferir_volumen(modelo, hu, spacing, cfg, dev, amp=cfg["entrenamiento"]["amp"], img01_256=x)
            lat.append(res["latencia"]["ms_por_corte"])

            reg_gt = region_de_etiqueta(gt)
            pres = np.stack([(reg_gt == r).reshape(gt.shape[0], -1).any(1) for r in (1, 2, 3)], 1)
            y_cls.append(pres.astype(int))
            p_cls.append(res["cls_prob"][:, :3])
            preds += res["detecciones"]
            gts += cajas_gt(gt)
            conf += matriz_confusion_pixel(reg_gt, region_de_etiqueta(res["etiquetas"]))

            mf = metricas_fragmentos(gt, res["etiquetas"], spacing, superficie=not args.sin_superficie)
            filas_frag += [{**f, "caso": c["id"]} for f in mf["detalle"]]
            tgt = tabla_fragmentos(gt, spacing)
            filas_dist += [{**f, "caso": c["id"]} for f in comparar_con_gt(res["tabla"], tgt, res["etiquetas"], gt)]
            por_caso.append({"caso": c["id"], "dice": mf["dice_fragmento"], "iou": mf["iou_fragmento"],
                             "hd95": mf["hd95_mm"]})
            print(f"[nivel {nivel}] {c['id']}  Dice {mf['dice_fragmento']:.3f}  IoU {mf['iou_fragmento']:.3f}"
                  f"  {res['latencia']['ms_por_corte']:.1f} ms/corte")

        err = [f["error_mm"] for f in filas_dist if f["error_mm"] is not None]
        err_c = [f["error_cabeza_mm"] for f in filas_dist if f["error_cabeza_mm"] is not None]
        resumen = {
            "clasificacion": metricas_clasificacion(np.concatenate(y_cls), np.concatenate(p_cls)),
            "deteccion": resumen_deteccion(preds, gts),
            "segmentacion": {
                "dice_fragmento": float(np.mean([f["dice"] for f in filas_frag])),
                "iou_fragmento": float(np.mean([f["iou"] for f in filas_frag])),
                "hd95_mm": _media_nan([f["hd95_mm"] for f in filas_frag]),
                "assd_mm": _media_nan([f["assd_mm"] for f in filas_frag]),
            },
            "distancia": {
                "n_fragmentos_secundarios_gt": len(filas_dist),
                "emparejados": len(err),
                "mae_geometrica_mm": float(np.mean(err)) if err else None,
                "mae_cabeza_mm": float(np.mean(err_c)) if err_c else None,
            },
            "matriz_confusion_pixel": conf.tolist(),
            "latencia_ms_por_corte": {"dispositivo": dev.type, "media": float(np.mean(lat))},
            "peores_5": sorted(por_caso, key=lambda d: d["dice"])[:5],
        }
        if nivel == 0 and len(calidad) >= 3:
            t = terciles({k: v["nitidez_lap"] for k, v in calidad.items()}, mayor_es_mejor=True)
            resumen["por_tercil_calidad"] = {
                n: float(np.mean([p["dice"] for p in por_caso if t[p["caso"]] == n] or [np.nan]))
                for n in ("alta", "media", "baja")}
        por_nivel[nivel] = resumen
        with open(dir_out / f"fragmentos_nivel{nivel}.json", "w", encoding="utf-8") as f:
            json.dump({"segmentacion": filas_frag, "distancia": filas_dist}, f, indent=1, default=float)

    salida = {"ckpt": str(args.ckpt), "split": args.split, "casos": sorted(ids), "calidad": calidad,
              "por_nivel_degradacion": por_nivel}
    with open(dir_out / f"evaluacion_{dev.type}.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=1, default=float)

    print("\n================ RESUMEN (nivel 0) ================")
    r0 = por_nivel[args.degradacion[0]]
    obj = [("F1 clasificacion", r0["clasificacion"]["f1_macro"], 0.85), ("AUC clasificacion", r0["clasificacion"]["auc_macro"], 0.85),
           ("IoU promedio bbox", r0["deteccion"]["iou_promedio"], 0.65), ("mAP@0.50", r0["deteccion"]["mAP@0.50"], 0.65),
           ("mAP@[0.50:0.95]", r0["deteccion"]["mAP@[0.50:0.95]"], 0.40), ("Dice fragmento", r0["segmentacion"]["dice_fragmento"], 0.85),
           ("IoU fragmento", r0["segmentacion"]["iou_fragmento"], 0.70)]
    for n, v, meta in obj:
        print(f"{n:22s} {v:7.3f}   meta >= {meta:.2f}   {'OK' if v >= meta else '--'}")
    print(f"HD95 {r0['segmentacion']['hd95_mm']:.2f} mm | ASSD {r0['segmentacion']['assd_mm']:.2f} mm")
    print(f"distancia MAE geometrica {r0['distancia']['mae_geometrica_mm']} mm | cabeza {r0['distancia']['mae_cabeza_mm']} mm")
    print(f"latencia {r0['latencia_ms_por_corte']['media']:.1f} ms/corte en {dev.type}")
    print(f"resultados en {dir_out}")


if __name__ == "__main__":
    main()
