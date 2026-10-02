"""Asignacion optima (algoritmo hungaro), implementada con NumPy.

Reemplaza a scipy.optimize.linear_sum_assignment. En Windows con Application Control,
scipy.optimize arrastra un DLL (_arpacklib) que la politica bloquea, y no se puede
importar. Aqui las matrices son diminutas (a lo sumo 10 x 10 fragmentos por region),
asi que un O(n^3) en Python puro sobra.

Una prueba compara el costo total contra scipy cuando scipy.optimize se puede importar.
"""
from __future__ import annotations

import numpy as np


def _hungaro_cuadrado(a: np.ndarray) -> np.ndarray:
    """Hungaro con potenciales (Kuhn-Munkres) sobre una matriz cuadrada k x k.
    Devuelve col_de_fila: la columna asignada a cada fila."""
    k = a.shape[0]
    inf = float("inf")
    u = np.zeros(k + 1)
    v = np.zeros(k + 1)
    p = np.zeros(k + 1, dtype=int)      # p[j] = fila asignada a la columna j (1-indexado)
    way = np.zeros(k + 1, dtype=int)
    for i in range(1, k + 1):
        p[0] = i
        j0 = 0
        minv = np.full(k + 1, inf)
        usado = np.zeros(k + 1, dtype=bool)
        while True:
            usado[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, k + 1):
                if not usado[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(k + 1):
                if usado[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col_de_fila = np.zeros(k, dtype=int)
    for j in range(1, k + 1):
        col_de_fila[p[j] - 1] = j - 1
    return col_de_fila


def asignacion_optima(costo) -> tuple[np.ndarray, np.ndarray]:
    """Minimiza la suma de costos asignando cada fila a una columna distinta.

    Misma convencion que scipy.optimize.linear_sum_assignment: acepta matrices
    rectangulares y devuelve (filas, columnas) de min(n, m) parejas, ordenadas por fila.
    Para MAXIMIZAR (p. ej. IoU) pasar -matriz.
    """
    c = np.asarray(costo, dtype=float)
    if c.ndim != 2:
        raise ValueError("la matriz de costos debe ser 2D")
    n, m = c.shape
    if n == 0 or m == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    if not np.isfinite(c).all():
        raise ValueError("la matriz de costos tiene valores no finitos")
    k = max(n, m)
    cuadrada = np.zeros((k, k))          # relleno constante: no altera el optimo de las parejas reales
    cuadrada[:n, :m] = c
    col = _hungaro_cuadrado(cuadrada)
    filas = np.array([i for i in range(n) if col[i] < m], dtype=int)
    return filas, col[filas]
