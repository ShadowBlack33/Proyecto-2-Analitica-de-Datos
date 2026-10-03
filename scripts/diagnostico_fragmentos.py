"""Diagnostico de la segmentacion de fragmentos y ajuste del post-proceso, sin reentrenar.

    python scripts/diagnostico_fragmentos.py --ckpt checkpoints/completo/mejor_conjunto.pth
    python scripts/diagnostico_fragmentos.py --ckpt ... --barrido

Trabaja sobre VALIDACION por defecto: los parametros del post-proceso se eligen aqui y se
aplican una sola vez sobre test. Elegirlos mirando test inflaria las metricas finales.

Paso 1 (cache): corre la red una vez por caso y guarda sus mapas a resolucion nativa,
recortados a la zona del hueso. Las corridas siguientes no vuelven a usar la GPU.

Paso 2 (diagnostico): para cada fragmento secundario del ground truth dice que le paso:
    correcto       lo cubre un solo fragmento predicho, sin mezclarse
    absorbido      la mayor parte quedo dentro del fragmento principal predicho (error de rol)
    partido        quedo repartido en 2 o mas fragmentos predichos (sobre-segmentacion)
    fusionado      su fragmento predicho tambien cubre otro secundario real (fusion)
    perdido        la mayor parte quedo como fondo u otra region
Ademas mide el Dice de la union de secundarios: si es alto, el modelo encuentra el hueso
fracturado y el problema es separar instancias; si es bajo, el problema es el rol.

Paso 3 (--barrido): prueba combinaciones de parametros del post-proceso sobre la cache y
recomienda la mejor por Dice de fragmento promedio.

Salida en resultados/<nombre>/diagnostico_<split>/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.etiquetas import indice_en_region, region_de_etiqueta  # noqa: E402
from src.data.io import leer_volumen, listar_casos  # noqa: E402
from src.data.splits import cargar_splits  # noqa: E402
from src.eval.asignacion import asignacion_optima  # noqa: E402
from src.inferencia import cargar_modelo, inferir_volumen, rol_con_umbral  # noqa: E402
from src.postprocess.instancias import min_voxeles_efectivo, volumen_etiquetas  # noqa: E402
from src.utils import cargar_config, dispositivo, fijar_semilla  # noqa: E402

REGIONES = {1: "SA", 2: "LI", 3: "RI"}


def recorte(*mascaras, margen=3):
    m = np.zeros_like(mascaras[0], bool)
    for x in mascaras:
        m |= x
    idx = np.where(m)
    return tuple(slice(max(0, int(i.min()) - margen), int(i.max()) + 1 + margen) for i in idx)


def cachear(casos, modelo, cfg, dev, dir_cache, rehacer):
    dir_cache.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(casos, 1):
        ruta = dir_cache / f"{c['id']}.npz"
        if ruta.exists() and not rehacer:
            continue
        t0 = time.perf_counter()
        hu, spacing, _ = leer_volumen(c["imagen"])
        gt, _, _ = leer_volumen(c["etiqueta"])
        res = inferir_volumen(modelo, hu, spacing, cfg, dev, amp=cfg["entrenamiento"]["amp"], devolver_mapas=True)
        m = res["mapas"]
        sl = recorte(gt > 0, m["sem"] > 0)
        np.savez_compressed(
            ruta, gt=gt[sl].astype(np.uint8), sem=m["sem"][sl], rol=m["rol"][sl],
            psec=np.clip(m["psec"][sl].astype(np.float32) * 255, 0, 255).astype(np.uint8),
            borde=np.clip(m["borde"][sl].astype(np.float32) * 255, 0, 255).astype(np.uint8),
            etq=res["etiquetas"][sl], spacing=np.array(spacing, np.float32))
        del hu, gt, res, m
        print(f"[cache {i}/{len(casos)}] {c['id']}  {time.perf_counter() - t0:.1f}s")


def cargar(ruta):
    d = np.load(ruta)
    return {k: d[k] for k in d.files} | {"psec": d["psec"].astype(np.float32) / 255,
                                         "borde": d["borde"].astype(np.float32) / 255,
                                         "spacing": tuple(float(v) for v in d["spacing"])}


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else 1.0


def dice_fragmentos(gt, etq):
    """Dice por fragmento del GT con emparejamiento hungaro por region (igual que evaluar.py)."""
    filas = []
    for r in (1, 2, 3):
        g_ids = [int(v) for v in np.unique(gt) if v > 0 and region_de_etiqueta(v) == r]
        p_ids = [int(v) for v in np.unique(etq) if v > 0 and region_de_etiqueta(v) == r]
        if not g_ids:
            continue
        mat = np.zeros((len(g_ids), max(1, len(p_ids))))
        for i, g in enumerate(g_ids):
            mg = gt == g
            for j, p in enumerate(p_ids):
                mat[i, j] = dice(mg, etq == p)
        fil, col = asignacion_optima(-mat)
        par = {g_ids[i]: mat[i, j] for i, j in zip(fil, col) if p_ids and mat[i, j] > 0}
        for g in g_ids:
            filas.append({"gt": int(g), "principal": indice_en_region(g) == 1, "dice": float(par.get(g, 0.0)),
                          "detectado": g in par})
    return filas


def categorizar(gt, etq, spacing):
    """Que le paso a cada fragmento secundario del ground truth."""
    vox_ml = float(np.prod(spacing)) / 1000.0
    filas = []
    for g in [int(v) for v in np.unique(gt) if v > 0 and indice_en_region(v) >= 2]:
        r = int(region_de_etiqueta(g))
        base = (r - 1) * 10
        mg = gt == g
        n = int(mg.sum())
        vals, cuentas = np.unique(etq[mg], return_counts=True)
        frac = dict(zip(vals.tolist(), (cuentas / n).tolist()))
        f_fondo = sum(f for v, f in frac.items() if v == 0 or region_de_etiqueta(v) != r)
        f_principal = frac.get(base + 1, 0.0)
        sec = {v: f for v, f in frac.items() if v > 0 and region_de_etiqueta(v) == r and indice_en_region(v) >= 2}
        partes = [v for v, f in sec.items() if f >= 0.2]
        dominante = max(sec, key=sec.get) if sec else None

        if f_fondo >= 0.5:
            cat = "perdido"
        elif f_principal >= 0.5:
            cat = "absorbido"
        elif len(partes) >= 2:
            cat = "partido"
        else:
            cat = "correcto"
            if dominante is not None:
                mp = etq == dominante
                otros = [int(v) for v in np.unique(gt[mp]) if v > 0 and v != g and indice_en_region(v) >= 2]
                if any((gt[mp] == o).sum() >= 0.2 * mp.sum() for o in otros):
                    cat = "fusionado"
        d = dice(mg, etq == dominante) if dominante is not None else 0.0
        filas.append({"gt": g, "region": REGIONES[r], "volumen_ml": n * vox_ml, "categoria": cat,
                      "frac_fondo": f_fondo, "frac_principal": f_principal, "n_partes": len(partes),
                      "dice_mejor_pred": d})
    return filas


def union_secundarios(gt, sem, rol, etq):
    """Dice de la union de secundarios por region: rol crudo de la red y despues del post-proceso."""
    filas = []
    for r in (1, 2, 3):
        g = (region_de_etiqueta(gt) == r) & (indice_en_region(gt) >= 2)
        if not g.any():
            continue
        crudo = (sem == r) & (rol == 2)
        post = (region_de_etiqueta(etq) == r) & (indice_en_region(etq) >= 2)
        filas.append({"region": REGIONES[r], "dice_rol_crudo": dice(g, crudo), "dice_post": dice(g, post)})
    return filas


def postproceso(d, cfg, modo, umbral, min_vox, usar_borde, erosion):
    rol = d["rol"] if umbral is None else rol_con_umbral(d["psec"], umbral)
    return volumen_etiquetas(d["sem"], rol, d["borde"].astype(np.float32), d["spacing"], min_vox,
                             cfg["postproceso"]["max_fragmentos_por_region"], usar_borde, erosion, modo)


def contar_pred(etq):
    return int(sum(1 for v in np.unique(etq) if v > 0))


def resumen_dice(filas, n_pred=None):
    df = pd.DataFrame(filas)
    sec = df[~df["principal"]]
    r = {"dice_fragmento": df["dice"].mean(), "dice_principal": df[df["principal"]]["dice"].mean(),
         "dice_secundario": sec["dice"].mean(), "secundarios_detectados": sec["detectado"].mean()}
    if n_pred is not None:
        tp = int(df["detectado"].sum())
        r.update({"fragmentos_pred": n_pred, "fragmentos_gt": len(df), "espurios": n_pred - tp,
                  "precision_fragmentos": tp / n_pred if n_pred else float("nan")})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default=None)
    ap.add_argument("--nombre", default=None)
    ap.add_argument("--max_casos", type=int, default=None)
    ap.add_argument("--rehacer_cache", action="store_true")
    ap.add_argument("--barrido", action="store_true", help="prueba combinaciones del post-proceso")
    ap.add_argument("--umbrales", nargs="*", type=float, default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--min_vox", nargs="*", type=int, default=[100, 200])
    ap.add_argument("--erosiones", nargs="*", type=int, default=[1])
    ap.add_argument("--sin_modo_borde", action="store_true", help="no probar la separacion solo por borde")
    ap.add_argument("--barrido_tamano", action="store_true", help="prueba tamanos minimos de fragmento en mL")
    ap.add_argument("--min_ml", nargs="*", type=float, default=[0.0, 0.5, 1.0, 2.0, 3.0])
    args = ap.parse_args()

    cfg = cargar_config(args.config)
    fijar_semilla(cfg["seed"])
    R = cfg["rutas"]
    ids = set(cargar_splits(R["splits"])[args.split])
    casos = [c for c in listar_casos(R["imagenes"], R["etiquetas"]) if c["id"] in ids][: args.max_casos]
    nombre = args.nombre or Path(args.ckpt).parent.name
    dir_out = Path(R["resultados"]) / nombre / f"diagnostico_{args.split}"
    dir_cache = dir_out / "cache"

    faltan = [c for c in casos if args.rehacer_cache or not (dir_cache / f"{c['id']}.npz").exists()]
    if faltan:
        dev = dispositivo(args.device)
        modelo = cargar_modelo(args.ckpt, cfg, dev)
        cachear(faltan, modelo, cfg, dev, dir_cache, args.rehacer_cache)
        del modelo

    # ------------------------------------------------ diagnostico con el post-proceso actual
    cat, uni, frag = [], [], []
    n_pred_actual = 0
    for c in casos:
        d = cargar(dir_cache / f"{c['id']}.npz")
        cat += [{**f, "caso": c["id"]} for f in categorizar(d["gt"], d["etq"], d["spacing"])]
        uni += [{**f, "caso": c["id"]} for f in union_secundarios(d["gt"], d["sem"], d["rol"], d["etq"])]
        frag += [{**f, "caso": c["id"]} for f in dice_fragmentos(d["gt"], d["etq"])]
        n_pred_actual += contar_pred(d["etq"])
    cat, uni = pd.DataFrame(cat), pd.DataFrame(uni)
    dir_out.mkdir(parents=True, exist_ok=True)
    cat.to_csv(dir_out / "secundarios.csv", index=False)
    uni.to_csv(dir_out / "union_secundarios.csv", index=False)

    pd.set_option("display.width", 140)
    r0 = resumen_dice(frag, n_pred_actual)
    print(f"\n==== DIAGNOSTICO ({args.split}, {len(casos)} casos, post-proceso actual) ====")
    print(f"Dice fragmento {r0['dice_fragmento']:.3f} | principales {r0['dice_principal']:.3f} | "
          f"secundarios {r0['dice_secundario']:.3f} | secundarios detectados {r0['secundarios_detectados']:.0%}")
    print(f"Fragmentos: {r0['fragmentos_pred']} predichos / {r0['fragmentos_gt']} reales | "
          f"precision {r0['precision_fragmentos']:.3f} | {r0['espurios']} espurios")

    print("\nQue le paso a cada fragmento secundario real:")
    tabla = cat.groupby("categoria").agg(fragmentos=("gt", "size"), volumen_ml_mediana=("volumen_ml", "median"),
                                          dice_medio=("dice_mejor_pred", "mean"))
    tabla["porcentaje"] = tabla["fragmentos"] / tabla["fragmentos"].sum()
    print(tabla.sort_values("fragmentos", ascending=False).round(3).to_string())

    print("\nDice de la union de secundarios (sin separar instancias):")
    print(f"  rol crudo de la red : {uni['dice_rol_crudo'].mean():.3f}")
    print(f"  despues del post-proceso: {uni['dice_post'].mean():.3f}")

    cat["tamano"] = pd.qcut(cat["volumen_ml"], 3, labels=["pequeno", "mediano", "grande"], duplicates="drop")
    print("\nPor tamano del fragmento secundario (terciles de volumen):")
    print(cat.groupby("tamano", observed=True).agg(fragmentos=("gt", "size"), ml_max=("volumen_ml", "max"),
                                                    dice_medio=("dice_mejor_pred", "mean")).round(3).to_string())

    print("\nLectura:")
    print("  union alta + muchos 'partido'/'fusionado'  -> el problema es separar instancias (post-proceso)")
    print("  union baja + muchos 'absorbido'            -> el problema es el rol (red, requiere reentrenar)")
    print("  muchos 'perdido' pequenos                  -> bajar min_voxeles_fragmento")

    resumen = {"split": args.split, "casos": [c["id"] for c in casos], "actual": r0,
               "categorias": tabla.reset_index().to_dict(orient="records"),
               "union_rol_crudo": uni["dice_rol_crudo"].mean(), "union_post": uni["dice_post"].mean()}

    # ------------------------------------------------ barrido del post-proceso
    if args.barrido:
        pp = cfg["postproceso"]
        combos = [("rol", None, pp["min_voxeles_fragmento"], pp.get("usar_borde", True), pp.get("erosion_nucleo", 1))]
        combos += [("rol", u, mv, True, er) for u, mv, er in itertools.product(args.umbrales, args.min_vox, args.erosiones)]
        if not args.sin_modo_borde:
            combos += [("borde", None, mv, True, er) for mv, er in itertools.product(args.min_vox, [0, 1])]
        print(f"\n==== BARRIDO: {len(combos)} combinaciones x {len(casos)} casos (un caso a la vez) ====")
        por_combo = [[] for _ in combos]
        for i, c in enumerate(casos, 1):
            t0 = time.perf_counter()
            d = cargar(dir_cache / f"{c['id']}.npz")
            for k, (mo, u, mv, ub, er) in enumerate(combos):
                por_combo[k] += dice_fragmentos(d["gt"], postproceso(d, cfg, mo, u, mv, ub, er))
            del d
            print(f"[caso {i}/{len(casos)}] {c['id']}  {time.perf_counter() - t0:.0f}s")
        filas = [{"modo": mo, "umbral_secundario": u, "min_voxeles": mv, "usar_borde": ub, "erosion": er,
                  **resumen_dice(fr)} for (mo, u, mv, ub, er), fr in zip(combos, por_combo)]
        bar = pd.DataFrame(filas)
        bar.to_csv(dir_out / "barrido.csv", index=False)
        orden = bar.sort_values(["dice_fragmento", "dice_secundario"], ascending=False)
        print("\nMejores 8 combinaciones (la primera fila del CSV es el post-proceso actual):")
        print(orden.head(8).round(3).to_string(index=False))
        mejor = orden.iloc[0]
        base = bar.iloc[0]
        print(f"\nActual: Dice {base['dice_fragmento']:.3f} -> mejor: {mejor['dice_fragmento']:.3f} "
              f"(+{mejor['dice_fragmento'] - base['dice_fragmento']:.3f}) en {args.split}")
        print("\nPara aplicarlo, en configs/default.yaml -> postproceso:")
        print(f"  modo_instancias: {mejor['modo']}")
        print(f"  umbral_secundario: {'null' if pd.isna(mejor['umbral_secundario']) else mejor['umbral_secundario']}")
        print(f"  min_voxeles_fragmento: {int(mejor['min_voxeles'])}")
        print(f"  usar_borde: {str(bool(mejor['usar_borde'])).lower()}")
        print(f"  erosion_nucleo: {int(mejor['erosion'])}")
        print("Luego se evalua UNA vez sobre test con scripts/evaluar.py.")
        resumen["barrido_mejor"] = mejor.to_dict()

    # ------------------------------------------------ barrido del tamano minimo de fragmento
    if args.barrido_tamano:
        pp = cfg["postproceso"]
        base = (pp.get("modo_instancias", "rol"), pp.get("umbral_secundario"), pp["min_voxeles_fragmento"],
                pp.get("usar_borde", True), pp.get("erosion_nucleo", 1))
        vols = cat["volumen_ml"]
        print(f"\n==== BARRIDO DE TAMANO MINIMO: {args.min_ml} mL x {len(casos)} casos ====")
        print(f"Secundarios reales en {args.split}: el mas pequeno {vols.min():.2f} mL, percentil 10 {vols.quantile(0.1):.2f} mL")
        por = [[] for _ in args.min_ml]
        npr = [0] * len(args.min_ml)
        for i, c in enumerate(casos, 1):
            t0 = time.perf_counter()
            d = cargar(dir_cache / f"{c['id']}.npz")
            for k, ml in enumerate(args.min_ml):
                mv = min_voxeles_efectivo(base[2], ml, d["spacing"])
                etq = postproceso(d, cfg, base[0], base[1], mv, base[3], base[4])
                por[k] += dice_fragmentos(d["gt"], etq)
                npr[k] += contar_pred(etq)
            del d
            print(f"[caso {i}/{len(casos)}] {c['id']}  {time.perf_counter() - t0:.0f}s")
        tam = pd.DataFrame([{"min_volumen_ml": ml, **resumen_dice(fr, n)} for ml, fr, n in zip(args.min_ml, por, npr)])
        tam.to_csv(dir_out / "barrido_tamano.csv", index=False)
        print(tam.round(3).to_string(index=False))
        ref = tam.loc[tam["min_volumen_ml"].idxmin(), "dice_fragmento"]
        aceptables = tam[tam["dice_fragmento"] >= ref - 0.005]
        mejor = aceptables.sort_values(["precision_fragmentos", "dice_fragmento"], ascending=False).iloc[0]
        print(f"\nCriterio: la mayor precision sin perder mas de 0.005 de Dice de fragmento.")
        print(f"Recomendado: min_volumen_ml = {mejor['min_volumen_ml']} -> precision "
              f"{mejor['precision_fragmentos']:.3f}, Dice {mejor['dice_fragmento']:.3f}, {int(mejor['espurios'])} espurios")
        print("Para aplicarlo, en configs/default.yaml -> postproceso:")
        print(f"  min_volumen_ml: {'null' if mejor['min_volumen_ml'] == 0 else mejor['min_volumen_ml']}")
        resumen["barrido_tamano_mejor"] = mejor.to_dict()

    with open(dir_out / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1, default=float)
    print(f"\nresultados en {dir_out}")


if __name__ == "__main__":
    main()
