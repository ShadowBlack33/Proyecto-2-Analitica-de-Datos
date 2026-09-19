# Model Card — [nombre del modelo/version]

> **Advertencia:** este sistema tiene fines exclusivamente academicos. No es un dispositivo
> medico, no ha sido validado clinicamente y no debe usarse para apoyar decisiones
> quirurgicas reales.

## Arquitectura
- Backbone: FundidoraPC extendida (N bloques, X->Y canales) + CBAM
- Cabezas: clasificacion anatomica, deteccion (grid + NMS propio), segmentacion de
  instancia (primary/secondary + watershed), regresion de distancia (mm)
- Parametros entrenables: ...
- Transfer learning: [si/no, en que parte, de que pesos]

## Datos de entrenamiento
- Dataset: PENGWIN Task 1 (CT), Zenodo 10.5281/zenodo.10927452
- N casos entrenamiento / validacion / test: ...
- Splits: ...

## Metricas obtenidas
| Tarea | Metrica | Objetivo | Obtenido |
|---|---|---|---|
| Clasificacion | F1-score | >=0.85 | |
| Clasificacion | AUC | >=0.85 | |
| Deteccion | IoU promedio | >=0.65 | |
| Deteccion | mAP@0.50 | >=0.65 | |
| Deteccion | mAP@[0.50:0.95] | >=0.40 | |
| Segmentacion | Dice | >=0.85 | |
| Segmentacion | IoU | >=0.70 | |
| Segmentacion (extra) | HD95 | — | |
| Segmentacion (extra) | ASSD | — | |
| Medicion de distancia | error medio vs. GT (mm) | — | |

## Latencia
- GPU: ... ms/imagen
- CPU: ... ms/imagen

## Alcances
- ...

## Limitaciones conocidas
- Fragmentos que entran en contacto en la superficie de fractura pueden fusionarse por
  errores de segmentacion (ver watershed en `src/postprocess/`).
- ...

## Estudio de ablacion
| Configuracion | Metrica principal |
|---|---|
| Completo (con CBAM, con transfer learning) | |
| Sin CBAM | |
| Sin transfer learning en el backbone | |
