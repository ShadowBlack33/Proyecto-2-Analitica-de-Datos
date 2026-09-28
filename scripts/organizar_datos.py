"""Descomprime los 3 zip de Zenodo en la estructura que espera el codigo.

    python scripts/organizar_datos.py --zips ..              # los zip estan en la carpeta de afuera
    python scripts/organizar_datos.py --zips .. --md5        # ademas verifica la integridad de la descarga

Resultado:
    data/raw/images/  001.mha ... (part1 + part2 JUNTAS)
    data/raw/labels/  001.mha ... (aparte: se llaman igual que las imagenes)

Aplana cualquier subcarpeta que traigan los zip, no sobrescribe lo que ya existe
(se puede correr de nuevo si se corto a la mitad) y al final verifica que cada imagen
tenga su etiqueta. data/ esta en .gitignore: nada de esto se sube a GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

ZIPS = {
    # nombre del zip -> (carpeta destino, md5 publicado en Zenodo)
    "PENGWIN_CT_train_images_part1.zip": ("images", "1d7cd8dc537ba12c61165f1be3e86022"),
    "PENGWIN_CT_train_images_part2.zip": ("images", "308a619a9c24fbcfbd197127c4cb39fb"),
    "PENGWIN_CT_train_labels.zip": ("labels", "77b3a29d528c719a917e6f90eb9b20d4"),
}
EXT = (".mha", ".mhd", ".raw", ".nii", ".nii.gz")


def md5(ruta: Path, bloque: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def extraer(zip_path: Path, destino: Path) -> tuple[int, int]:
    destino.mkdir(parents=True, exist_ok=True)
    nuevos = saltados = 0
    with zipfile.ZipFile(zip_path) as z:
        miembros = [m for m in z.infolist() if not m.is_dir() and m.filename.lower().endswith(EXT)
                    and "__MACOSX" not in m.filename]
        for i, m in enumerate(miembros, 1):
            salida = destino / Path(m.filename).name          # aplana subcarpetas
            if salida.exists() and salida.stat().st_size == m.file_size:
                saltados += 1
                continue
            with z.open(m) as src, open(salida, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
            nuevos += 1
            print(f"\r  {i}/{len(miembros)} {salida.name:20s}", end="", flush=True)
    print()
    return nuevos, saltados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", default="..", help="carpeta donde estan los 3 zip de Zenodo")
    ap.add_argument("--destino", default=str(RAIZ / "data" / "raw"))
    ap.add_argument("--md5", action="store_true", help="verifica el md5 de cada zip (tarda ~1 min)")
    args = ap.parse_args()

    dir_zips = Path(args.zips).resolve()
    destino = Path(args.destino)
    faltan = [n for n in ZIPS if not (dir_zips / n).exists()]
    if faltan:
        sys.exit(f"No encontre en {dir_zips}: {faltan}")

    for nombre, (sub, md5_ok) in ZIPS.items():
        zp = dir_zips / nombre
        print(f"{nombre}  ({zp.stat().st_size / 1024**3:.2f} GB) -> {destino / sub}")
        if args.md5:
            h = md5(zp)
            if h != md5_ok:
                sys.exit(f"  md5 NO coincide ({h}): la descarga esta corrupta, vuelve a bajar este zip")
            print("  md5 OK")
        nuevos, saltados = extraer(zp, destino / sub)
        print(f"  {nuevos} extraidos, {saltados} ya existian")

    imgs = {p.name for p in (destino / "images").iterdir() if p.name.endswith(EXT)}
    labs = {p.name for p in (destino / "labels").iterdir() if p.name.endswith(EXT)}
    print(f"\nimagenes: {len(imgs)} | etiquetas: {len(labs)}")
    sin_lbl, sin_img = sorted(imgs - labs), sorted(labs - imgs)
    if sin_lbl:
        print(f"[aviso] imagenes sin etiqueta: {sin_lbl[:10]}")
    if sin_img:
        print(f"[aviso] etiquetas sin imagen: {sin_img[:10]}")
    if not sin_lbl and not sin_img:
        print("Todo emparejado. Ya pueden borrar (o respaldar) los zip y correr:\n"
              "  python scripts/preparar_datos.py\n  python scripts/eda.py")


if __name__ == "__main__":
    main()
