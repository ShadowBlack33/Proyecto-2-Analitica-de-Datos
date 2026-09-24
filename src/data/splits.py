"""Particion train/val/test POR PACIENTE (nunca por corte).

Los cortes vecinos de un CT son casi identicos: si cortes del mismo escaneo caen en
train y en test, el modelo "ve" el test durante el entrenamiento y las metricas salen
infladas. Se estratifica por nivel de fractura para que los tres splits tengan casos
faciles y dificiles, y se versiona con el SHA-256 del manifiesto.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..utils import sha256_json


def _estrato(res: dict) -> int:
    """0 = sin fracturas secundarias, 1 = pocas (1-3), 2 = muchas (4+)."""
    n_sec = sum(1 for e in res.get("etiquetas", []) if (e - 1) % 10 + 1 >= 2)
    return 0 if n_sec == 0 else (1 if n_sec <= 3 else 2)


def crear_splits(resumenes: list[dict], proporciones=(0.7, 0.15, 0.15), seed=42) -> dict:
    rng = random.Random(seed)
    por_estrato: dict[int, list[str]] = {}
    for r in resumenes:
        por_estrato.setdefault(_estrato(r), []).append(r["id"])
    splits = {"train": [], "val": [], "test": []}
    for ids in por_estrato.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_tr = round(n * proporciones[0])
        n_va = round(n * proporciones[1])
        splits["train"] += ids[:n_tr]
        splits["val"] += ids[n_tr:n_tr + n_va]
        splits["test"] += ids[n_tr + n_va:]
    splits = {k: sorted(v) for k, v in splits.items()}
    assert not (set(splits["train"]) & set(splits["test"])), "fuga de datos entre train y test"
    assert not (set(splits["train"]) & set(splits["val"])), "fuga de datos entre train y val"
    return {"seed": seed, "proporciones": list(proporciones), "splits": splits,
            "sha256": sha256_json(splits)}


def guardar_splits(obj: dict, ruta) -> None:
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def cargar_splits(ruta) -> dict:
    with open(ruta, encoding="utf-8") as f:
        obj = json.load(f)
    if sha256_json(obj["splits"]) != obj["sha256"]:
        raise ValueError("El manifiesto de splits fue modificado: el hash no coincide")
    return obj["splits"]
