# PENGWIN — deteccion, segmentacion y medicion de fracturas pelvicas en CT

Proyecto integrador Corte 2 — Analitica de Datos (modulo IA, Carlos Andres Ferro), UAO, 2026-2.

Sistema propio (sin frameworks de deteccion/segmentacion de alto nivel preentrenados) que
detecta cada region osea pelvica en un corte de CT, segmenta y clasifica los fragmentos
individuales por hueso, mide la distancia de separacion de cada fragmento conminuto en mm,
y lo integra en un dashboard con 3 visualizadores.

**Alcance academico.** Este sistema no es un dispositivo medico, no ha sido validado
clinicamente y no debe usarse para decisiones quirurgicas reales.

Dataset: [PENGWIN](https://pengwin.grand-challenge.org/) (MICCAI 2024), Task 1 (CT),
150 escaneos. Ver `docs/` para el plan completo y la bitacora de decisiones del equipo.

## Estructura

```
src/
  data/          carga NIfTI, ventaneo HU, splits, EDA de calidad de imagen
  models/        FundidoraPC extendida, CBAM, las 4 cabezas
  losses/        perdida compuesta multitarea
  postprocess/   NMS propio, watershed + distance_transform_edt
  eval/          IoU, Dice, mAP, HD95, ASSD, latencia GPU/CPU
  dashboard/     los 3 visualizadores (MIP raw, inferencia corte-a-corte, reconstruccion 3D)
notebooks/       EDA y experimentos exploratorios
scripts/         entrenamiento, evaluacion, despliegue
checkpoints/     pesos entrenados (.pth) — no versionados, ver .gitignore
docs/            model card, plan del proyecto, bitacora
```

## Arquitectura

Backbone `FundidoraPC` extendida (5 bloques, 32->512 canales) + CBAM antes de bifurcar en
4 cabezas: clasificacion anatomica, deteccion (grid + NMS propio), segmentacion de instancia
(primary/secondary + watershed), regresion de distancia en mm.

Prohibido: YOLO, Detectron2, Mask R-CNN preentrenado como arquitectura entregada. SAM solo
como baseline zero-shot de comparacion.

## Reproducir

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
```

Semillas fijas y comandos exactos de cada etapa: ver `scripts/`.

## Flujo de trabajo del equipo

- La rama `main` esta protegida — todo cambio entra por Pull Request con revision por pares.
- Decisiones y su motivo quedan registradas en la bitacora del equipo (`docs/bitacora.md`
  o el artefacto publicado, ver enlace en `docs/`).
- Uso de IA generativa documentado en `IA_USAGE.md`.

## Cronograma

| Semana | Entregable |
|---|---|
| 8 | Dataset curado, carga NIfTI + ventaneo HU, EDA, visualizador 1 |
| 9 | Backbone + CBAM + deteccion, overfit test |
| 10 | Pipeline completo, medicion de distancia, comparacion SAM, metricas |
| 11 | Visualizadores 2 y 3, dashboard con tunel de Cloudflare, latencia, model card, pitch |
