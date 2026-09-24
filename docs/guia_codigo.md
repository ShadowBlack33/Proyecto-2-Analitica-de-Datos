# Guía del código — PENGWIN

Documento interno del equipo (no es el informe). Explica qué hace cada archivo, qué
requisito del enunciado cubre y qué decisiones de diseño quedan por validar con datos reales.
Todo lo que se diga en la sustentación debe poder señalarse en uno de estos archivos.

## Flujo por semanas

| Semana | Comando | Qué produce |
|---|---|---|
| 8 | `python scripts/preparar_datos.py` | caché de cortes 256×256 + `data/splits.json` (por paciente, con SHA-256) |
| 8 | `python scripts/eda.py` | `resultados/eda/*.csv` + gráficas: spacing, fragmentos por región, distancias GT, calidad |
| 8 | `streamlit run src/dashboard/app.py` (pestaña 1) | visualizador 1 (MIP raw) — no necesita modelo |
| 9 | `python scripts/perfil_memoria.py` | VRAM pico por batch/resolución (correr en la RTX 3050) |
| 9 | `python scripts/overfit_test.py --cabezas cls det` | prueba de correctitud: la pérdida debe caer >90 % |
| 9 | `python scripts/calibrar_lambdas.py` | tabla de magnitudes y gradientes → lambdas justificados |
| 9-10 | `python scripts/train.py --nombre completo` | checkpoints por etapa + `log_entrenamiento.csv` |
| 10 | `python scripts/evaluar.py --ckpt ... --degradacion 0 1 2 3` | todas las métricas de la sección 5 + robustez |
| 10 | `python scripts/baseline_sam.py --ckpt ... --sam ...` | comparación SAM vs modelo propio |
| 11 | `python scripts/evaluar.py --ckpt ... --device cpu` | latencia en CPU (la de GPU sale de la corrida anterior) |
| 11 | `streamlit run src/dashboard/app.py` + `cloudflared tunnel --url http://localhost:8501` | demo en vivo |

Antes de tener los datos reales, todo el pipeline se puede probar con CT sintéticos:
```bash
python scripts/generar_datos_sinteticos.py --n 8 --salida data/sintetico
# apuntar rutas.imagenes / rutas.etiquetas de la config a data/sintetico/images y /labels
```

## Dónde vive cada requisito

| Requisito del enunciado | Archivo |
|---|---|
| Taxonomía exacta (SA / coxal izq / coxal der, ≤10 fragmentos) | `src/data/etiquetas.py` |
| Spacing desde el header, nunca píxeles | `src/data/io.py` (lee `.mha`; el dataset NO viene en `.nii`) |
| Ventaneo HU | `src/data/preprocesamiento.py::ventanear` |
| Splits fijos por paciente | `src/data/splits.py` |
| Backbone FundidoraPC extendida + transfer learning solo en backbone | `src/models/fundidora.py` |
| CBAM antes de bifurcar | `src/models/cbam.py`, bloques 4 y 5 |
| Cabezas desde cero (clasificación, detección grid, segmentación de instancia) + 4ª de distancia | `src/models/modelo.py` |
| NMS propio | `src/postprocess/nms.py` (validado contra torchvision en `tests/`) |
| Dos etapas: región → fragmentos | `src/postprocess/instancias.py` |
| Distancia en mm con `distance_transform_edt`, también sobre el GT | `src/postprocess/distancia.py` |
| Pérdida compuesta con λ justificados | `src/losses/perdidas.py` + `scripts/calibrar_lambdas.py` |
| AMP, GPU y CPU, latencia | `src/entrenamiento.py`, `src/inferencia.py` |
| Métricas (F1, AUC, IoU, mAP, Dice, IoU fragmento, HD95, ASSD) | `src/eval/` |
| 3 visualizadores con la distancia visible en 2 y 3 | `src/dashboard/visualizadores.py`, `app.py` |
| Advertencia de uso no clínico | parte superior de `app.py` y `docs/model_card_template.md` |

## Decisiones de diseño (y cómo defenderlas)

**Entrada 2.5D (3 cortes vecinos como canales).** El enunciado pide 2D corte a corte
"manteniendo el fundamento 3D". Tres cortes consecutivos dan contexto volumétrico local
sin el costo de una red 3D, y encajan con backbones de 3 canales.

**Detección anchor-free sobre un grid de 16×16.** Cada celda predice clase + distancias a los
4 bordes (idea de FCOS/YOLOv8, implementación propia). Sin anclas que diseñar, y como la
taxonomía garantiza a lo sumo una región de cada tipo por corte, `max_det_por_clase = 1`.

**Segmentación = semántica (región) + rol (principal/secundario) + borde de contacto.**
Es el esquema primary-secondary del 2º lugar del challenge, más una clase de borde que es
la versión ligera del boundary-core del 1er lugar. Las instancias se separan en 3D
(componentes conexas + watershed sembrado con núcleos sin borde).

**Cuarta cabeza: distancia densa.** En vez de un número por fragmento (que exigiría saber
de antemano cuántos hay), predice para cada píxel secundario `log(1 + mm)` al fragmento
principal. El mínimo dentro de un fragmento es la distancia borde a borde. Así se comparan
tres cosas: GT, medición geométrica sobre la máscara predicha, y la cabeza.
*Pendiente de confirmar con el profesor que esta es la "cabeza de regresión" que pidió.*

**Clasificación como filtro.** La cabeza de clasificación mira todo el corte (pooling
promedio + máximo) y apaga las regiones que declara ausentes (`postproceso.filtrar_con_clasificador`).
Si en la validación resulta que empeora, se desactiva y se reporta en la ablación.

**Curriculum de entrenamiento.** detección → +segmentación → +distancia (backbone
congelado) → ajuste conjunto con lr ×0.1. Cada etapa arranca desde el mejor checkpoint de
validación de la anterior.

**Robustez a imágenes de baja calidad.** Aumentos de degradación (ruido, blur, pérdida de
resolución, gamma) con probabilidad 0.5 en entrenamiento; en evaluación, niveles fijos
0–3 y terciles de calidad real (varianza del Laplaciano).

**Flip horizontal desactivado por defecto.** Voltear un corte convierte el coxal izquierdo
en derecho. Si se activa, `aumentos.py` intercambia las etiquetas 11–20 ↔ 21–30.

## Cosas para verificar apenas lleguen los datos reales

- Que las etiquetas estén en 0..30 y que 1/11/21 sean el fragmento principal (`eda.py` avisa).
- Orientación: `io.py` reorienta todo a LPS; revisar en el visualizador 2 que el coxal
  izquierdo del paciente quede siempre del mismo lado de la imagen.
- Ventana HU `[-200, 1200]` y umbral del MIP (200 HU): ajustarlos con los percentiles que imprime `eda.py`.
- Tamaño de la caché en disco (~40–80 MB por caso) y tiempo por caso de `preparar_datos.py`.
- VRAM real con `perfil_memoria.py` en la RTX 3050 antes de fijar `batch_size`.
- Piso de resolución de la distancia: fragmentos en contacto dan ~1 vóxel (0.5–1 mm), no 0.
