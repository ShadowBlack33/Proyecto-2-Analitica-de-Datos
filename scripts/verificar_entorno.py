"""Revisa que el entorno pueda importar todo lo que usa el proyecto y que haya GPU.

    python scripts/verificar_entorno.py

En Windows con Smart App Control / Application Control, algunos archivos compilados
(.pyd/.dll) de librerias se bloquean al cargarlos. Este script lo detecta antes de
que falle un entrenamiento a mitad de camino.
"""
import importlib
import sys

MODULOS = [
    ("torch", "entrenamiento"), ("torchvision", "entrenamiento"),
    ("numpy", "todo"), ("scipy.ndimage", "EDT, watershed, distancias"),
    ("scipy.optimize", "emparejamiento de fragmentos"),
    ("SimpleITK", "lectura de .mha"), ("skimage.measure", "marching cubes"),
    ("skimage.segmentation", "watershed"), ("yaml", "configuracion"),
    ("pandas", "EDA y notebook"), ("matplotlib", "graficas"),
    ("plotly", "visualizadores"), ("streamlit", "dashboard"),
]

fallas = 0
for mod, uso in MODULOS:
    try:
        importlib.import_module(mod)
        print(f"  OK       {mod:22s} {uso}")
    except Exception as e:
        fallas += 1
        motivo = "BLOQUEADO por Windows" if "Application Control" in str(e) else type(e).__name__
        print(f"  FALLA    {mod:22s} {uso}  -> {motivo}: {str(e)[:80]}")

import torch  # noqa: E402

print(f"\nPython {sys.version.split()[0]} | PyTorch {torch.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} ({p.total_memory / 1024**3:.1f} GB)")
else:
    print("SIN GPU: PyTorch quedo en version CPU o no hay driver NVIDIA")

print("\nEntorno listo." if fallas == 0 else f"\n{fallas} modulo(s) con problemas: revisar antes de entrenar.")
