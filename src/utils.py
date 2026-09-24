"""Utilidades transversales: semilla, configuracion, dispositivo y huellas SHA-256."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def fijar_semilla(seed: int = 42) -> None:
    """Semilla unica para Python, NumPy y PyTorch (requisito de reproducibilidad)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cargar_config(ruta: str | Path = "configs/default.yaml") -> dict:
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dispositivo(preferido: str | None = None) -> torch.device:
    """GPU si existe; el pipeline debe correr igual en CPU (requisito del enunciado)."""
    if preferido:
        return torch.device(preferido)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True).encode("utf-8"))


def huella_tensor(t: torch.Tensor, decimales: int = 4) -> str:
    """Huella de predicciones sobre un lote fijo: demuestra que el modelo evaluado
    es el mismo que se entrega (se guarda en el checkpoint y en el model card)."""
    arr = np.round(t.detach().float().cpu().numpy(), decimales)
    return sha256_bytes(arr.tobytes())


def contar_parametros(modelo: torch.nn.Module, solo_entrenables: bool = True) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad or not solo_entrenables)
