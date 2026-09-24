"""Casos de control a mano, misma disciplina de S6/S7: validar cada pieza antes de confiar en ella.

    python -m pytest tests -q        (o: python tests/test_basicos.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.aumentos import _swap_lateralidad  # noqa: E402
from src.data.etiquetas import mapa_borde, region_de_etiqueta, rol_de_etiqueta  # noqa: E402
from src.eval.metricas_det import ap_11_puntos, iou_cajas  # noqa: E402
from src.eval.metricas_seg_cls import dice_iou, distancias_superficie  # noqa: E402
from src.losses.perdidas import asignar_targets, decodificar_cajas, giou  # noqa: E402
from src.postprocess.distancia import tabla_fragmentos  # noqa: E402
from src.postprocess.instancias import volumen_etiquetas  # noqa: E402
from src.postprocess.nms import nms_manual  # noqa: E402


def test_iou_casos_control():
    assert iou_cajas([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_cajas([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(iou_cajas([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-9


def test_giou_identico_es_uno():
    a = torch.tensor([[0.0, 0, 10, 10]])
    assert torch.allclose(giou(a, a), torch.ones(1), atol=1e-5)


def test_nms_contra_torchvision():
    torch.manual_seed(0)
    xy = torch.rand(60, 2) * 200
    wh = torch.rand(60, 2) * 60 + 5
    cajas = torch.cat([xy, xy + wh], 1)
    scores = torch.rand(60)
    propio = nms_manual(cajas, scores, 0.5)
    try:
        from torchvision.ops import nms
    except Exception:  # torchvision sin operadores compilados: se omite la comparacion
        return
    ref = nms(cajas, scores, 0.5)
    assert torch.equal(propio.sort().values, ref.sort().values)


def test_taxonomia():
    lbl = np.array([0, 1, 2, 10, 11, 15, 20, 21, 30])
    assert region_de_etiqueta(lbl).tolist() == [0, 1, 1, 1, 2, 2, 2, 3, 3]
    assert rol_de_etiqueta(lbl).tolist() == [0, 1, 2, 2, 1, 2, 2, 1, 2]


def test_flip_intercambia_lateralidad():
    lbl = torch.tensor([0, 1, 11, 15, 21, 29])
    assert _swap_lateralidad(lbl).tolist() == [0, 1, 21, 25, 11, 19]


def test_borde_solo_en_contacto():
    L = np.zeros((10, 10), np.uint8)
    L[2:8, 2:5] = 11
    L[2:8, 5:8] = 12
    b = mapa_borde(L)
    assert b[:, 4].sum() > 0 and b[:, 5].sum() > 0     # columnas en contacto
    assert b[:, 2].sum() == 0                           # borde con el fondo no cuenta


def test_asignacion_y_decodificacion():
    cajas = torch.tensor([[[40.0, 40, 120, 120], [0, 0, 0, 0], [0, 0, 0, 0]]])
    hay = torch.tensor([[True, False, False]])
    cls_t, ltrb_t, pos, caja_t = asignar_targets(cajas, hay, 16, 16, stride=16)
    assert pos.sum() > 0 and cls_t[0, 1].sum() == 0
    dec = decodificar_cajas(ltrb_t, 16).permute(0, 2, 3, 1)[pos]
    assert torch.allclose(dec, torch.tensor([40.0, 40, 120, 120]).expand_as(dec), atol=1e-4)


def test_ap_perfecto():
    assert abs(ap_11_puntos(np.array([1.0]), np.array([1.0])) - 1.0) < 1e-9


def test_distancia_en_mm_usa_spacing():
    etq = np.zeros((5, 20, 40), np.uint8)
    etq[1:4, 5:15, 2:10] = 21        # principal
    etq[1:4, 5:15, 20:25] = 22       # fragmento a 10 voxeles en x
    f = {r["nombre"]: r for r in tabla_fragmentos(etq, (2.0, 1.0, 0.5))}
    assert abs(f["RI-2"]["distancia_mm"] - 11 * 0.5) < 1e-6    # 11 voxeles entre centros x 0.5 mm


def test_postproceso_reconstruye_gt():
    etq = np.zeros((12, 30, 30), np.uint8)
    etq[2:10, 5:25, 5:15] = 11
    etq[2:10, 5:25, 15:20] = 12       # en contacto con el principal
    etq[2:10, 5:15, 22:28] = 13       # separado
    borde = np.stack([mapa_borde(e) for e in etq]).astype(np.float32)
    rec = volumen_etiquetas(region_de_etiqueta(etq), rol_de_etiqueta(etq), borde, (1, 1, 1), min_vox=10)
    assert sorted(np.unique(rec).tolist()) == [0, 11, 12, 13]
    d, j = dice_iou(rec > 0, etq > 0)
    assert d == 1.0


def test_hd95_identico_cero():
    m = np.zeros((10, 10, 10), bool)
    m[2:8, 2:8, 2:8] = True
    hd, assd = distancias_superficie(m, m, (1, 1, 1))
    assert hd == 0 and assd == 0


if __name__ == "__main__":
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print("ok", nombre)


def test_metricas_clasificacion_igual_a_sklearn():
    """F1, AUC y matriz de confusion propias == sklearn (si sklearn se puede importar)."""
    import pytest
    try:
        from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    except Exception:
        pytest.skip("sklearn no disponible (p. ej. bloqueado por Application Control)")
    from src.eval.metricas_seg_cls import auc_binaria, f1_por_clase, matriz_confusion

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (500, 3))
    prob = np.clip(y * 0.3 + rng.random((500, 3)) * 0.7, 0, 1).round(2)   # redondeo -> empates
    pred = prob >= 0.5
    assert np.allclose(f1_por_clase(y, pred), f1_score(y, pred, average=None, zero_division=0))
    for c in range(3):
        assert abs(auc_binaria(y[:, c], prob[:, c]) - roc_auc_score(y[:, c], prob[:, c])) < 1e-12
    a, b = rng.integers(0, 4, 1000), rng.integers(0, 4, 1000)
    assert (matriz_confusion(a, b, 4) == confusion_matrix(a, b, labels=[0, 1, 2, 3])).all()
    # caso borde: clase sin positivos ni predichos -> F1 0, como zero_division=0
    assert f1_por_clase(np.zeros((5, 1)), np.zeros((5, 1)))[0] == 0.0
