"""Lectura de volumenes CT y mascaras.

OJO: el dataset de Zenodo viene en .mha (MetaImage), no en .nii. El enunciado habla
del "header NIfTI" pero el principio es el mismo: el espaciado fisico (mm/voxel) se lee
del header del archivo, nunca se asume. SimpleITK lee .mha, .nii y .nii.gz igual.

Convencion de ejes en todo el proyecto: arreglo NumPy en orden (z, y, x) y
spacing en el MISMO orden (sz, sy, sx). SimpleITK entrega el spacing en (x, y, z),
por eso se invierte aqui y solo aqui.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

EXTENSIONES = (".mha", ".mhd", ".nii", ".nii.gz")


def _orientar(img: sitk.Image, orientacion: str = "LPS") -> sitk.Image:
    """Lleva todos los casos a la misma orientacion. Sin esto, un escaner que guarda
    el volumen volteado haria que 'izquierda' en la imagen no sea la izquierda del paciente
    y el modelo aprenderia la lateralidad al reves en esos casos."""
    try:
        return sitk.DICOMOrient(img, orientacion)
    except Exception:  # imagenes sin matriz de direccion valida
        return img


def leer_volumen(ruta: str | Path, orientar: bool = True):
    """Devuelve (array zyx, spacing zyx en mm, metadatos)."""
    img = sitk.ReadImage(str(ruta))
    if orientar:
        img = _orientar(img)
    arr = sitk.GetArrayFromImage(img)                 # (z, y, x)
    spacing = tuple(float(s) for s in img.GetSpacing()[::-1])  # (sz, sy, sx)
    meta = {
        "origen": img.GetOrigin(),
        "direccion": img.GetDirection(),
        "spacing_xyz": img.GetSpacing(),
        "forma_zyx": arr.shape,
    }
    return arr, spacing, meta


def id_caso(ruta: str | Path) -> str:
    nombre = Path(ruta).name
    for ext in EXTENSIONES:
        if nombre.endswith(ext):
            return nombre[: -len(ext)]
    return Path(ruta).stem


def listar_casos(dir_imagenes: str | Path, dir_etiquetas: str | Path | None = None) -> list[dict]:
    """Empareja imagen y etiqueta por id de caso (001.mha <-> 001.mha).
    Las imagenes de part1.zip y part2.zip deben quedar en la MISMA carpeta."""
    dir_imagenes = Path(dir_imagenes)
    imgs = {id_caso(p): p for p in sorted(dir_imagenes.rglob("*")) if p.name.endswith(EXTENSIONES)}
    labs = {}
    if dir_etiquetas is not None:
        labs = {id_caso(p): p for p in sorted(Path(dir_etiquetas).rglob("*")) if p.name.endswith(EXTENSIONES)}
    casos = []
    for cid, p in imgs.items():
        casos.append({"id": cid, "imagen": p, "etiqueta": labs.get(cid)})
    faltan = [c["id"] for c in casos if dir_etiquetas is not None and c["etiqueta"] is None]
    if faltan:
        print(f"[aviso] {len(faltan)} casos sin etiqueta: {faltan[:10]}")
    return casos


def guardar_volumen(arr: np.ndarray, spacing_zyx, ruta: str | Path, referencia: dict | None = None) -> None:
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in spacing_zyx[::-1]))
    if referencia is not None:
        img.SetOrigin(referencia["origen"])
        img.SetDirection(referencia["direccion"])
    sitk.WriteImage(img, str(ruta), useCompression=True)
