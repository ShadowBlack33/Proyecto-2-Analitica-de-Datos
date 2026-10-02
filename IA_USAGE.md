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

## Sesion — eliminacion de scikit-learn (bloqueo de Windows)

**Contexto:** en el PC de entrenamiento, Windows (Smart App Control / Application Control)
bloqueo el DLL `scipy.stats._rcont`. scikit-learn importa `scipy.stats` al cargar, asi que
`train.py` fallaba en la primera validacion.

**Que se obtuvo de la IA:** F1, AUC-ROC (formula de Mann-Whitney con empates promediados)
y matriz de confusion implementadas con NumPy en `src/eval/metricas_seg_cls.py`; se quito
scikit-learn de `requirements.txt`; script `scripts/verificar_entorno.py` para diagnosticar.

**Verificacion:** prueba `test_metricas_clasificacion_igual_a_sklearn` (resultados identicos
a sklearn, incluidos empates y clases vacias). Se simulo el bloqueo (import de `scipy.stats`
y `sklearn` fallando) y se corrieron pruebas, entrenamiento, evaluacion, inferencia y
dashboard: todo funciona y las metricas coinciden con la version anterior.

**Pendiente del equipo:** analisis critico y modificaciones manuales.

## Sesion — configuraciones de ablacion y transfer learning desde el Taller 3

**Que se obtuvo de la IA:** herencia de configuracion (`hereda:` en el YAML) para que cada
ablacion difiera de `default.yaml` en un solo factor; `configs/ablacion_sin_cbam.yaml` y
`configs/ablacion_con_tl.yaml`; carga de pesos del backbone reescrita para emparejar por capa
(conv con conv, BatchNorm con BatchNorm) y absorber el bias de la conv en la media del
BatchNorm; al reanudar a mitad de etapa se conserva el mejor puntaje ya alcanzado.

**Error detectado por la propia prueba:** la version anterior de la carga emparejaba tensores
en orden; si las convoluciones del Taller 3 tienen bias, el emparejamiento se desalineaba y
solo copiaba 5 tensores. Se corrigio y se agrego una prueba de equivalencia numerica.

**Verificacion:** 14 pruebas pasan; ambas ablaciones entrenan en datos sinteticos; la salida
del bloque copiado coincide con la original (diferencia maxima 5e-7).

**Pendiente del equipo:** analisis critico y modificaciones manuales.

## Sesion — cargador con varios procesos y segundo bloqueo de Windows

**Contexto:** (1) con `num_workers: 4` la ablacion con transfer learning fallo en la epoca 11
(`OSError 22`): en Windows cada epoca recreaba los procesos y les enviaba una copia del dataset con
los volumenes abiertos. (2) Windows bloqueo `scipy.optimize` (DLL `_arpacklib`), que se usaba solo para
emparejar fragmentos con el algoritmo hungaro.

**Que se obtuvo de la IA:** el dataset ya no envia los volumenes abiertos y los procesos de carga se crean
una sola vez (`persistent_workers`); algoritmo hungaro propio en `src/eval/asignacion.py`; el verificador de
entorno ahora tambien importa los modulos del proyecto, para detectar bloqueos indirectos; el cargador del
checkpoint del Taller 3 busca los pesos dentro de la clave `backbone`.

**Errores de la IA detectados:** la primera version del cargador de pesos asumia un solo diccionario y fallo
con el formato real del Taller 3 (`backbone`, `rpn`, `det`); `verificar_entorno.py` reporto `scipy.optimize`
como OK porque en ese momento aun no estaba bloqueado: la comprobacion por librerias sueltas no basta.

**Verificacion:** 16 pruebas pasan, entre ellas el hungaro propio contra scipy en 200 matrices aleatorias con
empates; con el bloqueo simulado corren entrenamiento, evaluacion y diagnostico; y las metricas por fragmento
son identicas a las de la version con scipy en 144 fragmentos (mismas parejas, mismo Dice).

**Pendiente del equipo:** analisis critico y modificaciones manuales.

## Sesion — interpretabilidad en el notebook

**Contexto:** el notebook no mostraba el gamma de BatchNorm de los bloques, Grad-CAM, como se ajustan las
cajas ni pacientes con fractura (solo se veian caderas sin fragmentos).

**Que se obtuvo de la IA:** `src/analisis.py` con gamma de BatchNorm por capa (y % de canales apagados),
Grad-CAM y Smooth Grad-CAM de la cabeza de clasificacion comparados con el mapa de objectness (correlacion y
masa dentro de la caja, como en el Taller 3), candidatos antes y despues del NMS, IoU de cajas en test y
seleccion de pacientes y cortes con fractura; nueva seccion 7 del notebook y seccion 6 sobre el paciente de
test con mas fragmentos.

**Verificacion:** el notebook se ejecuto completo en modo prueba sin errores; prueba automatica de que el
calculo del gamma detecta canales apagados. Con el modelo real, ningun canal del backbone esta apagado
(gamma medio entre 0.91 y 0.99 por bloque).

**Pendiente del equipo:** analisis critico y modificaciones manuales.
