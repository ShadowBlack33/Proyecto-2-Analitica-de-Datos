# Registro de uso de IA generativa

Este archivo documenta cada uso de IA generativa durante el proyecto, como lo exige la
sección 7 del enunciado. Una entrada por sesion/decision relevante — no hace falta registrar
cada mensaje suelto, pero si toda decision de arquitectura, codigo generado, o correccion
importante que haya surgido con apoyo de IA.

Formato de cada entrada:

## [Fecha] — [Tema corto]

**Modelo usado:** (ej. Claude Sonnet 4.6, ChatGPT, etc.)

**Prompt exacto (o resumen fiel si fue muy largo):**
> ...

**Que se obtuvo:**
- Codigo / decision / explicacion que vino de la IA.

**Que se modifico o verifico manualmente:**
- Que cambiaron ustedes despues, y por que.

**Analisis critico:**
- Que acerto la IA, que se equivoco o no aplicaba directamente, y como lo detectaron.

---

<!-- Agregar entradas nuevas abajo de esta linea -->

## 2026-09-23 — Primera version del codigo base (sin datos reales)

**Modelo usado:** Claude (Anthropic), Opus 5.5, en claude.ai.

**Prompt (resumen fiel):**
> "No te puedo pasar los datos porque son muy pesados, ya actualizamos la bitacora con cosas
> de este proyecto, puedes adelantar lo que mas se pueda del codigo por favor"
> (con el contexto previo: enunciado completo, bitacora de decisiones, hardware del equipo).

**Que se obtuvo:**
- Estructura completa de `src/` (datos, modelo, perdidas, post-proceso, metricas, inferencia,
  dashboard) y `scripts/` (preparar datos, EDA, overfit test, calibrar lambdas, entrenar,
  evaluar, baseline SAM, perfil de memoria, inferir, generador de CT sinteticos).
- Pruebas de control en `tests/` y guia del codigo en `docs/guia_codigo.md`.
- Correccion de un bug en `.gitignore` (`data/` tambien ignoraba `src/data/`).

**Que se verifico (por la IA, sobre CT sinteticos, en CPU):**
- `tests/test_basicos.py` pasa completo, incluido NMS propio == `torchvision.ops.nms`.
- Post-proceso alimentado con el GT reconstruye exactamente los fragmentos (Dice 1.0) y
  las distancias en mm.
- Overfit test con las 4 cabezas: la perdida cae y el IoU de cajas sobre el lote llega a ~0.87.
- `train.py` completo (4 etapas) y `evaluar.py` corren de punta a punta.
- Dashboard (`app.py`) ejecutado en modo headless con `streamlit.testing.AppTest`: las 4
  pestanas, los 3 visualizadores, el slider de cortes y la latencia funcionan sin excepciones.
- Evaluacion de un checkpoint entrenado solo hasta la etapa de segmentacion, sobre CT
  sinteticos (faciles, NO comparables con PENGWIN real): nivel 0 -> F1 1.00, mAP@0.50 0.95,
  Dice fragmento 0.75; nivel 3 de degradacion -> mAP@0.50 0.91, Dice fragmento 0.46.
  Sirve solo para confirmar que las metricas y el analisis de robustez reaccionan bien.

**Que NO se ha verificado (pendiente del equipo):**
- Nada se ha corrido con los datos reales de PENGWIN ni en GPU.
- La convencion de etiquetas (1/11/21 = principal) se tomo de la literatura del challenge;
  confirmarla con `scripts/eda.py`.
- La interpretacion de la "cabeza de regresion" como distancia densa esta pendiente de
  confirmar con el profesor.
- Los lambdas del config son provisionales hasta correr `calibrar_lambdas.py` con datos reales.
- `baseline_sam.py` no se pudo ejecutar en el entorno de la IA (requiere descargar los pesos de SAM).
- El tunel de Cloudflare no se probo; el dashboard si.

**Que se modifico manualmente:** (completar por el equipo al revisar y adaptar el codigo)

**Analisis critico:** (completar: que se entendio, que se cambio y por que)
