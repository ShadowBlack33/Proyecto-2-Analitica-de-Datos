"""Dashboard PENGWIN (Streamlit) — 3 visualizadores + tabla de fragmentos + latencia.

Local:
    streamlit run src/dashboard/app.py
Demo en vivo (seccion 7), desde otra terminal:
    cloudflared tunnel --url http://localhost:8501
    (cerrar con: taskkill /IM cloudflared.exe /F en Windows)
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.data.io import leer_volumen, listar_casos  # noqa: E402
from src.dashboard.visualizadores import vis_corte, vis_mip_raw, vis_reconstruccion  # noqa: E402
from src.utils import cargar_config, dispositivo  # noqa: E402

AVISO = ("**Uso exclusivamente académico.** Este sistema no es un dispositivo médico, no ha sido "
         "validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.")

st.set_page_config(page_title="PENGWIN · fracturas pélvicas", page_icon="🦴", layout="wide")
st.warning(AVISO, icon="⚠️")
st.title("🦴 PENGWIN — detección, segmentación y medición de fracturas pélvicas")

cfg = cargar_config(RAIZ / "configs" / "default.yaml")


@st.cache_resource(show_spinner="Cargando modelo…")
def _modelo(ruta: str, dev: str):
    from src.inferencia import cargar_modelo
    import torch
    return cargar_modelo(ruta, cfg, torch.device(dev))


@st.cache_data(show_spinner="Leyendo volumen…", max_entries=2)
def _volumen(ruta: str):
    return leer_volumen(ruta)


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("Entrada")
    fuente = st.radio("Volumen", ["Caso del dataset", "Subir archivo (.mha / .nii.gz)"])
    ruta_vol = ruta_gt = None
    if fuente == "Caso del dataset":
        casos = listar_casos(RAIZ / cfg["rutas"]["imagenes"], RAIZ / cfg["rutas"]["etiquetas"])
        if casos:
            ids = [c["id"] for c in casos]
            sel = st.selectbox("Caso", ids)
            c = casos[ids.index(sel)]
            ruta_vol, ruta_gt = str(c["imagen"]), (str(c["etiqueta"]) if c["etiqueta"] else None)
        else:
            st.info("No se encontraron casos en rutas.imagenes")
    else:
        up = st.file_uploader("CT", type=["mha", "nii", "gz"])
        if up is not None:
            suf = ".nii.gz" if up.name.endswith(".nii.gz") else Path(up.name).suffix
            tmp = Path(tempfile.gettempdir()) / f"pengwin_subido{suf}"
            tmp.write_bytes(up.getvalue())
            ruta_vol = str(tmp)

    st.header("Modelo")
    cks = sorted(str(p.relative_to(RAIZ)) for p in (RAIZ / cfg["rutas"]["checkpoints"]).rglob("mejor_*.pth"))
    ruta_ck = st.selectbox("Checkpoint", cks) if cks else None
    opciones_dev = ["cuda", "cpu"] if dispositivo().type == "cuda" else ["cpu"]
    dev = st.selectbox("Dispositivo", opciones_dev)
    correr = st.button("Ejecutar inferencia", type="primary", disabled=not (ruta_vol and ruta_ck))
    ver_gt = st.checkbox("Mostrar ground truth en el corte", value=False, disabled=ruta_gt is None)

if not ruta_vol:
    st.info("Elige un caso o sube un CT en la barra lateral.")
    st.stop()

hu, spacing, _ = _volumen(ruta_vol)
st.caption(f"Volumen {hu.shape[0]}×{hu.shape[1]}×{hu.shape[2]} vóxeles · spacing (z, y, x) = "
           f"{tuple(round(s, 3) for s in spacing)} mm — leído del header del archivo")

clave = f"res::{ruta_vol}::{ruta_ck}::{dev}"
if correr:
    import torch
    from src.inferencia import inferir_volumen
    t0 = time.perf_counter()
    res = inferir_volumen(_modelo(RAIZ / ruta_ck if not Path(ruta_ck).is_absolute() else ruta_ck, dev),
                          hu, spacing, cfg, torch.device(dev))
    res["latencia"]["total_s"] = time.perf_counter() - t0
    st.session_state[clave] = res
res = st.session_state.get(clave)

t1, t2, t3, t4 = st.tabs(["1 · Volumen crudo (MIP)", "2 · Inferencia corte a corte",
                          "3 · Reconstrucción 3D", "Fragmentos y latencia"])

with t1:
    c1, c2 = st.columns([1, 3])
    umbral = c1.slider("Umbral de hueso (HU)", 100, 700, int(cfg["datos"]["umbral_hueso_hu"]), 25)
    ang = c1.slider("Rotación (°)", -90, 90, 0, 15)
    c1.caption("Preprocesamiento clásico: no interviene el modelo.")
    c2.plotly_chart(vis_mip_raw(hu, spacing, umbral, ang), width="stretch")

if res is None:
    for t in (t2, t3, t4):
        t.info("Ejecuta la inferencia desde la barra lateral.")
    st.stop()

etq = res["etiquetas"]
gt = _volumen(ruta_gt)[0] if (ver_gt and ruta_gt) else None
with t2:
    con_hueso = np.where(etq.reshape(etq.shape[0], -1).max(1) > 0)[0]
    z_ini = int(con_hueso[len(con_hueso) // 2]) if len(con_hueso) else etq.shape[0] // 2
    z = st.slider("Corte axial", 0, etq.shape[0] - 1, z_ini)
    cols = st.columns(2 if gt is not None else 1)
    cols[0].plotly_chart(vis_corte(hu[z], etq[z], res["detecciones"][z], res["tabla"],
                                   cfg["datos"]["ventana_hu"], titulo=f"Predicción · corte {z}"), width="stretch")
    if gt is not None:
        from src.postprocess.distancia import tabla_fragmentos
        cols[1].plotly_chart(vis_corte(hu[z], gt[z], [], tabla_fragmentos(gt, spacing),
                                       cfg["datos"]["ventana_hu"], titulo=f"Ground truth · corte {z}"), width="stretch")

with t3:
    st.plotly_chart(vis_reconstruccion(etq, spacing, res["tabla"]), width="stretch")

with t4:
    st.subheader("Fragmentos detectados")
    st.dataframe([{k: f[k] for k in ("nombre", "es_principal", "volumen_ml", "distancia_mm", "distancia_cabeza_mm")}
                  for f in res["tabla"]], width="stretch")
    lat = res["latencia"]
    a, b, c = st.columns(3)
    a.metric("Latencia por corte", f"{lat['ms_por_corte']:.1f} ms", lat["dispositivo"].upper())
    b.metric("Cortes procesados", lat["n_cortes"])
    c.metric("Tiempo total", f"{lat.get('total_s', lat['modelo_s']):.1f} s")
    st.download_button("Descargar tabla (JSON)", json.dumps(res["tabla"], indent=1), "fragmentos.json")
