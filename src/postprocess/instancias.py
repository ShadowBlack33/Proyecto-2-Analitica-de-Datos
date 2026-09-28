"""De las salidas densas por corte a fragmentos 3D con etiquetas PENGWIN.

Se hace en 3D, despues de apilar las predicciones 2D a resolucion nativa: un fragmento
es un objeto volumetrico y su identidad tiene que ser consistente entre cortes.

Modo "rol" (por defecto), por region (SA / LI / RI):
  1. cada voxel de la region es "principal" o "secundario" segun la cabeza de rol.
  2. principal = componente conexa 3D mas grande de los voxeles "principal";
     las componentes sueltas restantes se tratan como secundarias.
  3. secundarios en contacto se separan con watershed sobre la transformada de
     distancia, sembrado con los nucleos (secundario sin borde de contacto predicho).
     Esta es la mitigacion de la limitacion conocida del model card: fragmentos que se
     tocan en la superficie de fractura y se fusionan.
  4. fragmentos diminutos (< min_voxeles) se funden con el vecino mas cercano.
  5. etiquetas finales: principal = base+1, secundarios base+2.. por tamano descendente,
     maximo 10 por region (taxonomia del dataset).

Modo "borde": ignora la cabeza de rol. Parte cada region por las superficies de contacto
que predice la cabeza de borde (watershed sembrado con la region sin borde) y llama
principal al fragmento mas grande. Evita que la red tenga que decidir cual fragmento es
el principal, que es donde falla cuando un secundario es casi del tamano del principal.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

ESTRUCTURA_3D = ndi.generate_binary_structure(3, 1)


def _caja(m: np.ndarray, margen: int = 2):
    idx = np.where(m)
    return tuple(slice(max(0, int(i.min()) - margen), int(i.max()) + 1 + margen) for i in idx)


def _fundir_pequenos(etq: np.ndarray, mascara: np.ndarray, min_vox: int) -> np.ndarray:
    """Reasigna los voxeles de fragmentos pequenos al fragmento vecino mas cercano."""
    ids, cuentas = np.unique(etq[etq > 0], return_counts=True)
    pequenos = ids[cuentas < min_vox]
    if len(pequenos) == 0 or len(pequenos) == len(ids):
        return etq
    etq = etq.copy()
    etq[np.isin(etq, pequenos)] = 0
    _, (iz, iy, ix) = ndi.distance_transform_edt(etq == 0, return_indices=True)
    relleno = etq[iz, iy, ix]
    etq[mascara & (etq == 0)] = relleno[mascara & (etq == 0)]
    return etq


def separar_region(region: np.ndarray, rol: np.ndarray, borde: np.ndarray | None,
                   spacing, min_vox=200, max_frag=10, usar_borde=True, erosion=1,
                   modo="rol") -> list[np.ndarray]:
    """Devuelve [mascara_principal, mascara_sec_1, ...] (ya recortadas al volumen completo)."""
    if region.sum() < min_vox:
        return []
    sl = _caja(region)
    reg = region[sl]
    if modo == "borde":
        etq = _separar_por_borde(reg, None if borde is None else borde[sl], spacing, min_vox, erosion)
        return _mascaras_finales(etq, reg, region, sl, max_frag)
    es_sec = reg & (rol[sl] == 2)
    es_pri = reg & ~es_sec

    cc, n = ndi.label(es_pri if es_pri.any() else reg, structure=ESTRUCTURA_3D)
    if n == 0:
        return []
    tam = ndi.sum(np.ones_like(cc), cc, index=np.arange(1, n + 1))
    principal = cc == (int(np.argmax(tam)) + 1)
    secundario = reg & ~principal

    etq = np.zeros(reg.shape, np.int32)
    etq[principal] = 1
    if secundario.sum() >= min_vox:
        nucleo = secundario
        if borde is not None and usar_borde:
            nucleo = secundario & (borde[sl] < 0.5)
        if erosion > 0:
            nucleo = ndi.binary_erosion(nucleo, structure=ESTRUCTURA_3D, iterations=erosion)
        semillas, n_sem = ndi.label(nucleo, structure=ESTRUCTURA_3D)
        if n_sem == 0:
            semillas, n_sem = ndi.label(secundario, structure=ESTRUCTURA_3D)
        edt = ndi.distance_transform_edt(secundario, sampling=spacing)
        sep = watershed(-edt, semillas, mask=secundario)
        etq[sep > 0] = sep[sep > 0] + 1
    etq = _fundir_pequenos(etq, reg, min_vox)
    return _mascaras_finales(etq, reg, region, sl, max_frag)


def _separar_por_borde(reg, borde, spacing, min_vox, erosion):
    """Instancias de una region sin usar el rol: 1 = la mas grande, 2.. el resto."""
    nucleo = reg if borde is None else reg & (borde < 0.5)
    if erosion > 0:
        nucleo = ndi.binary_erosion(nucleo, structure=ESTRUCTURA_3D, iterations=erosion)
    semillas, n = ndi.label(nucleo, structure=ESTRUCTURA_3D)
    if n == 0:
        semillas, n = ndi.label(reg, structure=ESTRUCTURA_3D)
    edt = ndi.distance_transform_edt(reg, sampling=spacing)
    sep = _fundir_pequenos(watershed(-edt, semillas, mask=reg), reg, min_vox)
    ids, cuentas = np.unique(sep[sep > 0], return_counts=True)
    etq = np.zeros(reg.shape, np.int32)
    for nuevo, k in enumerate(ids[np.argsort(-cuentas)], start=1):
        etq[sep == k] = nuevo
    return etq


def _mascaras_finales(etq, reg, region, sl, max_frag):
    # principal primero, luego secundarios por tamano; los que excedan max_frag se funden
    ids, cuentas = np.unique(etq[etq > 1], return_counts=True)
    orden = ids[np.argsort(-cuentas)]
    sobrantes = orden[max_frag - 1:]
    if len(sobrantes):
        etq[np.isin(etq, sobrantes)] = 0
        _, (iz, iy, ix) = ndi.distance_transform_edt(etq == 0, return_indices=True)
        vacio = reg & (etq == 0)
        etq[vacio] = etq[iz, iy, ix][vacio]
        orden = orden[: max_frag - 1]

    mascaras = []
    for k in [1] + list(orden):
        m = np.zeros(region.shape, bool)
        m[sl] = etq == k
        if m.any():
            mascaras.append(m)
    return mascaras


def volumen_etiquetas(sem: np.ndarray, rol: np.ndarray, borde: np.ndarray | None, spacing,
                      min_vox=200, max_frag=10, usar_borde=True, erosion=1, modo="rol") -> np.ndarray:
    """sem/rol (Z,H,W) a resolucion nativa -> volumen con etiquetas PENGWIN 0..30."""
    out = np.zeros(sem.shape, np.uint8)
    for r in (1, 2, 3):
        mascaras = separar_region(sem == r, rol, borde, spacing, min_vox, max_frag, usar_borde, erosion, modo)
        for i, m in enumerate(mascaras, start=1):
            out[m & (out == 0)] = (r - 1) * 10 + i
    return out
