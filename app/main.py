import pandas as pd
import numpy as np
import os
from pathlib import Path
import streamlit as st
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# --- CONFIGURACIÓ DE PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "cancer_endometri.csv"
MODEL_PATH = BASE_DIR / "models" / "model_v1.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler_v1.joblib"
FEATURES_PATH = BASE_DIR / "models" / "selected_features_v1.joblib"

# CONSTANTS MAPPING
COL_ID = "codigo_participante"
COL_GRUPO_RIESGO_DEFINITIVO = "grupo_de_riesgo_definitivo"
COL_AFECTACION_LINF = "afectacion_linf"
COL_ESTADIAJE_PRE = "estadiaje_pre_i"
COL_GRADO = "grado_histologi"
COL_INFILTRACION = "infiltracion_mi"
COL_IMC = "imc"
COL_FIGO = "FIGO2023"
COL_RECEP_EST = "recep_est_porcent"
COL_RECEP_PROG = "rece_de_Ppor"
COL_EDAD = "edad"
COL_TTO_QUIRURGICO = "tto_1_quirugico"
COL_TTO_SISTEMICO = "Tratamiento_sistemico_realizad"
COL_HISTO = "histo_defin"
COL_METASTA = "metasta_distan"

# --- Carregar model, scaler i features ---
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

@st.cache_data
def load_data(file_buffer=None):
    source = file_buffer if file_buffer else DATA_PATH
    if not source and not os.path.exists(DATA_PATH):
        return None
    try:
        if file_buffer:
            if file_buffer.name.endswith('.csv'):
                df = pd.read_csv(source)
            else:
                df = pd.read_excel(source)
        else:
            if str(source).endswith('.csv'):
                df = pd.read_csv(source)
            else:
                df = pd.read_excel(source)
        if COL_ID in df.columns:
            df[COL_ID] = df[COL_ID].astype(str)
        return df
    except Exception:
        return None

# Configuració de la pàgina
st.set_page_config(
    page_title="NEST - Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("NEST")
st.markdown("**NSMP Endometrial Stratification Tool**")

# Carregar artefactes del model
try:
    model, scaler, SELECTED_FEATURES = load_model_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error carregant el model: {e}")

# --- SIDEBAR: SIMULACIÓ BASE DE DADES HOSPITAL ---
with st.sidebar:
    st.header("📂 Base de Dades Hospital")
    uploaded_file = st.file_uploader("Carregar Llistat Pacients (CSV/XLSX)", type=["csv", "xlsx"])
    
    # Només carregar dades si hi ha fitxer pujat
    df = None
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        # Si falla la càrrega del fitxer pujat, fallback silenciós al path
        if df is None:
            df = load_data(None)  # Intenta carregar des del path
    
    patient_id_input = st.text_input("Buscar ID Pacient", placeholder="Ex: 12345")
    patient = None
    
    if df is not None:
        st.info(f"Dades carregades: {len(df)} pacients")
        if patient_id_input:
            row = df[df[COL_ID] == patient_id_input]
            if not row.empty:
                st.success(f"Pacient {patient_id_input} trobat!")
                patient = row.iloc[0]
                
                if st.button("📥 Importar dades pacients"):
                    def set_state(key, col_name, cast_type=None, min_val=None, max_val=None, options=None, mapping=None):
                        final_val = None
                        if col_name in df.columns:
                            raw_val = patient[col_name]
                            if pd.notna(raw_val):
                                val = raw_val
                                is_valid = True
                                if cast_type:
                                    try:
                                        val = cast_type(val)
                                    except:
                                        is_valid = False
                                if is_valid:
                                    if mapping:
                                        if val in mapping:
                                            val = mapping[val]
                                        else:
                                            is_valid = False
                                    if is_valid and (min_val is not None or max_val is not None):
                                        if min_val is not None and val < min_val: val = min_val
                                        if max_val is not None and val > max_val: val = max_val
                                    if is_valid and options is not None:
                                        if val not in options:
                                            is_valid = False
                                    if is_valid:
                                        final_val = val
                        st.session_state[key] = final_val

                    # Mapping i import
                    set_state("edad", COL_EDAD, int, min_val=18, max_val=120)
                    set_state("imc", COL_IMC, float, min_val=10.0, max_val=60.0)
                    set_state("recep_est", COL_RECEP_EST, float, min_val=0.0, max_val=100.0)
                    set_state("recep_prog", COL_RECEP_PROG, float, min_val=0.0, max_val=100.0)
                    
                    RISK_MAPPING = {1: "Riesgo bajo", 2: "Riesgo intermedio", 3: "Riesgo intermedio-alto", 4: "Riesgo alto", 5: "Avanzados"}
                    set_state("grupo_riesgo", COL_GRUPO_RIESGO_DEFINITIVO, cast_type=int, mapping=RISK_MAPPING, options=list(RISK_MAPPING.values()))
                    
                    GRADO_MAPPING = {1: "Bajo grado (G1-G2)", 2: "Alto grado (G3)"}
                    set_state("grado", COL_GRADO, cast_type=int, mapping=GRADO_MAPPING, options=list(GRADO_MAPPING.values()))

                    INFILTRACION_MAPPING = {0: "No infiltracion", 1: "Infiltracion miometrial <50%", 2: "Infiltracion miometrial >50%", 3: "Infiltracion serosa"}
                    set_state("infiltracion", COL_INFILTRACION, cast_type=int, mapping=INFILTRACION_MAPPING, options=list(INFILTRACION_MAPPING.values()))
                    
                    LINF_MAPPING = {0: "No", 1: "Si"}
                    set_state("afect_linf", COL_AFECTACION_LINF, cast_type=int, mapping=LINF_MAPPING, options=["No", "Si"])
                    
                    ESTADIAJE_MAPPING = {0: "Estadio I", 1: "Estadio II", 2: "Estadio III y IV"}
                    set_state("estadiaje_pre", COL_ESTADIAJE_PRE, cast_type=int, mapping=ESTADIAJE_MAPPING, options=list(ESTADIAJE_MAPPING.values()))
                    
                    SISTEMICO_MAPPING = {0: "No realizada", 1: "Dosis parcial", 2: "Dosis completa"}
                    set_state("tto_sistemico", COL_TTO_SISTEMICO, cast_type=int, mapping=SISTEMICO_MAPPING, options=list(SISTEMICO_MAPPING.values()))
                    
                    FIGO_MAPPING = {1: "IA1", 2: "IA2", 3: "IA3", 4: "IB", 5: "IC", 6: "IIA", 7: "IIB", 8: "IIC", 9: "IIIA", 10: "IIIB", 11: "IIIC", 12: "IVA", 13: "IVB", 14: "IVC"}
                    set_state("figo", COL_FIGO, cast_type=int, mapping=FIGO_MAPPING, options=list(FIGO_MAPPING.values()))

                    QUIRURGICO_MAPPING = {0: "No", 1: "Si"}
                    set_state("tto_quirurgico", COL_TTO_QUIRURGICO, cast_type=int, mapping=QUIRURGICO_MAPPING, options=["No", "Si"])
                    
                    HISTO_MAPPING = {1: "Hiperplasia con atipias", 2: "Carcinoma endometrioide", 3: "Carcinoma seroso", 4: "Carcinoma de celulas claras", 5: "Carcinoma Indiferenciado", 6: "Carcinoma mixto", 7: "Carcinoma escamoso", 8: "Carcinosarcoma", 9: "Otros"}
                    set_state("histo", COL_HISTO, cast_type=int, mapping=HISTO_MAPPING, options=list(HISTO_MAPPING.values()))
                    
                    METASTA_MAPPING = {0: "No", 1: "Si"}
                    set_state("metasta", COL_METASTA, cast_type=int, mapping=METASTA_MAPPING, options=["No", "Si"])
                    
                    st.rerun()
            else:
                st.warning("ID no trobat.")

# --- STATE INITIALIZATION ---
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prob" not in st.session_state:
    st.session_state.prob = None
if "shap_values" not in st.session_state:
    st.session_state.shap_values = None
if "input_data" not in st.session_state:
    st.session_state.input_data = None

def init_state():
    keys = ["edad", "imc", "grupo_riesgo", "estadiaje_pre", "histo", "grado", 
            "infiltracion", "figo", "metasta", "tto_quirurgico", "tto_sistemico", 
            "afect_linf", "recep_est", "recep_prog"]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None

init_state()

# --- PESTANYES ---
tab1, tab2 = st.tabs(["Dades del Pacient", "Resultats"])

# --- TAB 1: INPUTS ---
with tab1:
    st.header("Introducció de Dades Clíniques")
    st.info("**Nota:** Els camps que quedin en blanc seran **imputats automàticament** pel sistema durant la predicció.")

    # Helper per obtenir index d'un selectbox
    def get_selectbox_index(options, key):
        val = st.session_state.get(key)
        if val is not None and val in options:
            return options.index(val)
        return None

    with st.container(border=True):
        with st.form("patient_form_tabs"):
            col1, col2, col3 = st.columns(3, gap="medium")
            
            with col1:
                st.subheader("Dades Generals")
                st.number_input("Edat (edad)", min_value=18, max_value=120, 
                               value=st.session_state.get("edad"), key="edad", placeholder="Edat...")
                st.number_input("IMC (imc)", min_value=10.0, max_value=60.0, 
                               value=st.session_state.get("imc"), format="%.2f", key="imc", placeholder="IMC...")
                risk_opts = ["Riesgo bajo", "Riesgo intermedio", "Riesgo intermedio-alto", "Riesgo alto", "Avanzados"]
                st.selectbox("Grup de Risc Definitiu", risk_opts, key="grupo_riesgo", 
                            index=get_selectbox_index(risk_opts, "grupo_riesgo"), placeholder="Seleccionar...")
                estadiaje_opts = ["Estadio I", "Estadio II", "Estadio III y IV"]
                st.selectbox("Estadiaje Pre-quirúrgic", estadiaje_opts, key="estadiaje_pre", 
                            index=get_selectbox_index(estadiaje_opts, "estadiaje_pre"), placeholder="Seleccionar...")

            with col2:
                st.subheader("Histologia i Tumor")
                histo_opts = ["Hiperplasia con atipias", "Carcinoma endometrioide", "Carcinoma seroso", "Carcinoma de celulas claras", "Carcinoma Indiferenciado", "Carcinoma mixto", "Carcinoma escamoso", "Carcinosarcoma", "Otros"]
                st.selectbox("Tipus Histològic", histo_opts, key="histo", 
                            index=get_selectbox_index(histo_opts, "histo"), placeholder="Seleccionar...")
                grado_opts = ["Bajo grado (G1-G2)", "Alto grado (G3)"]
                st.selectbox("Grau Histològic", grado_opts, key="grado", 
                            index=get_selectbox_index(grado_opts, "grado"), placeholder="Seleccionar...")
                infil_opts = ["No infiltracion", "Infiltracion miometrial <50%", "Infiltracion miometrial >50%", "Infiltracion serosa"]
                st.selectbox("Infiltració Miometrial", infil_opts, key="infiltracion", 
                            index=get_selectbox_index(infil_opts, "infiltracion"), placeholder="Seleccionar...")
                figo_opts = ["IA1", "IA2", "IA3", "IB", "IC", "IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA", "IVB", "IVC"]
                st.selectbox("Estadi FIGO 2023", figo_opts, key="figo", 
                            index=get_selectbox_index(figo_opts, "figo"), placeholder="Seleccionar...")
                yesno_opts = ["No", "Si"]
                st.selectbox("Metàstasi a Distància", yesno_opts, key="metasta", 
                            index=get_selectbox_index(yesno_opts, "metasta"), placeholder="Seleccionar...")

            with col3:
                st.subheader("Tractament i Altres")
                st.selectbox("Tractament Quirúrgic 1ari", yesno_opts, key="tto_quirurgico", 
                            index=get_selectbox_index(yesno_opts, "tto_quirurgico"), placeholder="Seleccionar...")
                sistemico_opts = ["No realizada", "Dosis parcial", "Dosis completa"]
                st.selectbox("Tractament Sistèmic Realitzat", sistemico_opts, key="tto_sistemico", 
                            index=get_selectbox_index(sistemico_opts, "tto_sistemico"), placeholder="Seleccionar...")
                st.selectbox("Afectació Limfàtica (LVSI)", yesno_opts, key="afect_linf", 
                            index=get_selectbox_index(yesno_opts, "afect_linf"), placeholder="Seleccionar...")
                st.number_input("Receptors Estrogen (%)", 0.0, 100.0, 
                               value=st.session_state.get("recep_est"), key="recep_est", placeholder="0-100%")
                st.number_input("Receptors Progesterona (%)", 0.0, 100.0, 
                               value=st.session_state.get("recep_prog"), key="recep_prog", placeholder="0-100%")
            
            st.markdown("---")
            submitted = st.form_submit_button("Calcular Risc de Recurrència", use_container_width=True, type="primary")

    if submitted and model_loaded:
        # Mapejats inversos per convertir UI -> valors numèrics
        RISK_INV = {"Riesgo bajo": 1, "Riesgo intermedio": 2, "Riesgo intermedio-alto": 3, "Riesgo alto": 4, "Avanzados": 5}
        GRADO_INV = {"Bajo grado (G1-G2)": 1, "Alto grado (G3)": 2}
        INFIL_INV = {"No infiltracion": 0, "Infiltracion miometrial <50%": 1, "Infiltracion miometrial >50%": 2, "Infiltracion serosa": 3}
        LINF_INV = {"No": 0, "Si": 1}
        ESTAD_INV = {"Estadio I": 0, "Estadio II": 1, "Estadio III y IV": 2}
        SIST_INV = {"No realizada": 0, "Dosis parcial": 1, "Dosis completa": 2}
        FIGO_INV = {"IA1": 1, "IA2": 2, "IA3": 3, "IB": 4, "IC": 5, "IIA": 6, "IIB": 7, "IIC": 8, "IIIA": 9, "IIIB": 10, "IIIC": 11, "IVA": 12, "IVB": 13, "IVC": 14}
        QUIR_INV = {"No": 0, "Si": 1}
        HISTO_INV = {"Hiperplasia con atipias": 1, "Carcinoma endometrioide": 2, "Carcinoma seroso": 3, "Carcinoma de celulas claras": 4, "Carcinoma Indiferenciado": 5, "Carcinoma mixto": 6, "Carcinoma escamoso": 7, "Carcinosarcoma": 8, "Otros": 9}
        META_INV = {"No": 0, "Si": 1}
        
        # =================================================================
        # IMPUTACIÓ REPLICANT EXACTAMENT EL NOTEBOOK 02_data_preprocessing
        # =================================================================
        
        # FASE 1: Primer determinem el grau histològic (necessari per imputar receptors)
        # Moda del dataset = 1 (Bajo grado G1-G2)
        grado = GRADO_INV.get(st.session_state.grado, 1)  # Moda = 1
        
        # FASE 6: Medianes estratificades per grau histològic (del notebook)
        # recep_est_porcent: {1.0: 90.0, 2.0: 70.0}
        # rece_de_Ppor: {1.0: 90.0, 2.0: 25.0}
        MEDIAN_RECEP_EST = {1: 90.0, 2: 70.0}
        MEDIAN_RECEP_PROG = {1: 90.0, 2: 25.0}
        
        # Imputar receptors segons grau
        if st.session_state.recep_est is not None:
            recep_est = st.session_state.recep_est
        else:
            recep_est = MEDIAN_RECEP_EST.get(grado, 90.0)
            
        if st.session_state.recep_prog is not None:
            recep_prog = st.session_state.recep_prog
        else:
            recep_prog = MEDIAN_RECEP_PROG.get(grado, 90.0)
        
        # Construir el DataFrame amb les features i imputació del notebook
        input_dict = {
            # FASE 9: grupo_de_riesgo_definitivo -> moda = 1
            "grupo_de_riesgo_definitivo": RISK_INV.get(st.session_state.grupo_riesgo, 1),
            
            # FASE 7: afectacion_linf -> 0 (assumim No si no registrat)
            "afectacion_linf": LINF_INV.get(st.session_state.afect_linf, 0),
            
            # FASE 9: estadiaje_pre_i -> moda = 0
            "estadiaje_pre_i": ESTAD_INV.get(st.session_state.estadiaje_pre, 0),
            
            # FASE 11: Tratamiento_sistemico_realizad -> 0 (No realizada)
            "Tratamiento_sistemico_realizad": SIST_INV.get(st.session_state.tto_sistemico, 0),
            
            # FASE 1: grado_histologi -> moda = 1
            "grado_histologi": grado,
            
            # FASE 9: infiltracion_mi -> moda = 1
            "infiltracion_mi": INFIL_INV.get(st.session_state.infiltracion, 1),
            
            # FASE 6: imc -> mediana global = 29.4
            "imc": st.session_state.imc if st.session_state.imc else 29.4,
            
            # FASE 9: FIGO2023 -> moda = 1
            "FIGO2023": FIGO_INV.get(st.session_state.figo, 1),
            
            # FASE 6: recep_est_porcent -> mediana estratificada per grau
            "recep_est_porcent": recep_est,
            
            # FASE 6: rece_de_Ppor -> mediana estratificada per grau
            "rece_de_Ppor": recep_prog,
            
            # edad -> sense imputació per defecte, usem mediana del dataset si cal
            "edad": st.session_state.edad if st.session_state.edad else 65,
            
            # FASE 11: tto_1_quirugico -> 1 (Sí, la majoria passen per cirurgia)
            "tto_1_quirugico": QUIR_INV.get(st.session_state.tto_quirurgico, 1),
            
            # FASE 9: histo_defin -> moda = 2 (Carcinoma endometrioide)
            "histo_defin": HISTO_INV.get(st.session_state.histo, 2),
            
            # FASE 7: metasta_distan -> 0 (assumim No si no registrat)
            "metasta_distan": META_INV.get(st.session_state.metasta, 0),
        }
        
        # Ordenar segons SELECTED_FEATURES
        input_df = pd.DataFrame([{f: input_dict[f] for f in SELECTED_FEATURES}])
        
        # Escalar
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=SELECTED_FEATURES)
        
        # Predicció
        prob = model.predict_proba(input_scaled_df)[0][1]
        st.session_state.prob = prob
        st.session_state.input_data = input_scaled_df
        
        # SHAP amb dades de fons (optimitzat per velocitat)
        with st.spinner("Calculant interpretabilitat..."):
            try:
                bg_data_path = BASE_DIR / "data" / "processed" / "preprocessed_v1.csv"
                if bg_data_path.exists():
                    bg_df = pd.read_csv(bg_data_path)
                    X_bg = bg_df[SELECTED_FEATURES]
                    X_bg_scaled = scaler.transform(X_bg)
                    # Reduït per accelerar
                    sample_size = min(20, len(X_bg_scaled))
                    background = X_bg_scaled[:sample_size]
                    
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_vals = explainer.shap_values(input_scaled, nsamples=50)
                    # Agafar la classe positiva (index 1)
                    st.session_state.shap_values = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
                else:
                    st.session_state.shap_values = None
            except Exception as e:
                st.session_state.shap_values = None
        
        st.session_state.prediction_done = True
        
        # Missatge indicant que vagi a Resultats
        st.success("Càlcul completat! Ves a la pestanya **Resultats** per veure l'anàlisi.")

# --- TAB 2: RESULTATS ---
with tab2:
    st.header("Resultats de l'Anàlisi")
    
    if not st.session_state.prediction_done or st.session_state.prob is None:
        st.warning("Primer has d'introduir les dades i calcular el risc a la pestanya anterior.")
    else:
        prob = st.session_state.prob
        
        # Semàfor de risc
        if prob < 0.30:
            risk_level = "Baix"
            color = "#00c853"
            emoji = "🟢"
        elif prob < 0.60:
            risk_level = "Moderat"
            color = "#ffab00"
            emoji = "🟡"
        else:
            risk_level = "Alt"
            color = "#ff1744"
            emoji = "🔴"
        
        col_pred, col_semaforo = st.columns([1, 1])
        
        with col_pred:
            st.markdown("### Probabilitat de Recurrència")
            st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 80px;'>{prob:.1%}</h1>", unsafe_allow_html=True)
        
        with col_semaforo:
            st.markdown("### Nivell de Risc")
            st.markdown(f"<h1 style='text-align: center; font-size: 60px;'>{emoji}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: {color};'>{risk_level}</h2>", unsafe_allow_html=True)
        
        st.divider()
        
        # --- SHAP ---
        st.subheader("Interpretabilitat - SHAP")
        if st.session_state.shap_values is not None and st.session_state.input_data is not None:
            shap_vals = st.session_state.shap_values
            input_data = st.session_state.input_data
            
            try:
                if isinstance(shap_vals, np.ndarray):
                    shap_flat = shap_vals.flatten()
                else:
                    shap_flat = np.array(shap_vals).flatten()
                
                n_features = len(SELECTED_FEATURES)
                if len(shap_flat) >= n_features:
                    shap_flat = shap_flat[:n_features]
                else:
                    shap_flat = np.pad(shap_flat, (0, n_features - len(shap_flat)), 'constant')
                
                # Noms llegibles per les features
                FEATURE_DISPLAY_NAMES = {
                    "grupo_de_riesgo_definitivo": "Grup de Risc",
                    "afectacion_linf": "LVSI",
                    "estadiaje_pre_i": "Estadiatge Pre",
                    "Tratamiento_sistemico_realizad": "Tto. Sistèmic",
                    "grado_histologi": "Grau Histològic",
                    "infiltracion_mi": "Infiltració MI",
                    "imc": "IMC",
                    "FIGO2023": "FIGO",
                    "recep_est_porcent": "Recep. Estrogen",
                    "rece_de_Ppor": "Recep. Progest.",
                    "edad": "Edat",
                    "tto_1_quirugico": "Tto. Quirúrgic",
                    "histo_defin": "Histologia",
                    "metasta_distan": "Metàstasi"
                }
                
                shap_df = pd.DataFrame({
                    "Feature": [FEATURE_DISPLAY_NAMES.get(f, f) for f in SELECTED_FEATURES],
                    "SHAP Value": shap_flat,
                })
                
                # Filtrar variables amb contribució significativa (>1% del màxim)
                max_abs = shap_df["SHAP Value"].abs().max()
                threshold = max_abs * 0.01  # 1% del valor màxim
                shap_df = shap_df[shap_df["SHAP Value"].abs() > threshold]
                
                # Agafar fins a top 10 més significatives
                shap_df = shap_df.sort_values("SHAP Value", key=abs, ascending=False).head(10)
                # Re-ordenar de menor a major per barh (els de baix apareixen a dalt)
                shap_df = shap_df.sort_values("SHAP Value", key=abs, ascending=True)
                
                n_vars = len(shap_df)
                fig_height = max(4, n_vars * 0.5)
                fig, ax = plt.subplots(figsize=(10, fig_height))
                colors = ["#ff1744" if v > 0 else "#00c853" for v in shap_df["SHAP Value"]]
                bars = ax.barh(range(len(shap_df)), shap_df["SHAP Value"].values, color=colors, height=0.6)
                ax.set_yticks(range(len(shap_df)))
                ax.set_yticklabels(shap_df["Feature"].values)
                ax.set_xlabel("Impacte en la probabilitat de recurrència")
                ax.set_title(f"Top {n_vars} Variables Més Significatives (SHAP)")
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax.tick_params(axis='y', labelsize=11)
                # Expandir eix x per veure totes les barres
                max_val = max(abs(shap_df["SHAP Value"].max()), abs(shap_df["SHAP Value"].min()))
                if max_val > 0:
                    ax.set_xlim(-max_val * 1.3, max_val * 1.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption("🔴 Vermell = Augmenta el risc | 🟢 Verd = Redueix el risc")
            except Exception as e:
                st.warning(f"Error visualitzant SHAP: {e}")
        else:
            st.info("No s'han pogut calcular els valors SHAP.")
        
        # --- PDP ---
        st.divider()
        st.subheader("Partial Dependence Plot (PDP)")
        
        # Definir variables categòriques vs contínues
        CATEGORICAL_FEATURES = {
            "grupo_de_riesgo_definitivo": {1: "Baix", 2: "Intermedi", 3: "Int-Alt", 4: "Alt", 5: "Avançat"},
            "afectacion_linf": {0: "No", 1: "Sí"},
            "estadiaje_pre_i": {0: "Estadi I", 1: "Estadi II", 2: "Estadi III-IV"},
            "Tratamiento_sistemico_realizad": {0: "No", 1: "Parcial", 2: "Completa"},
            "grado_histologi": {1: "Baix (G1-G2)", 2: "Alt (G3)"},
            "infiltracion_mi": {0: "No", 1: "<50%", 2: ">50%", 3: "Serosa"},
            "tto_1_quirugico": {0: "No", 1: "Sí"},
            "metasta_distan": {0: "No", 1: "Sí"},
        }
        
        CONTINUOUS_FEATURES = ["imc", "recep_est_porcent", "rece_de_Ppor", "edad"]
        
        # Noms llegibles per PDP
        PDP_FEATURE_NAMES = {
            "grupo_de_riesgo_definitivo": "Grup de Risc",
            "afectacion_linf": "LVSI",
            "estadiaje_pre_i": "Estadiatge Pre-quirúrgic",
            "Tratamiento_sistemico_realizad": "Tractament Sistèmic",
            "grado_histologi": "Grau Histològic",
            "infiltracion_mi": "Infiltració Miometrial",
            "imc": "IMC",
            "FIGO2023": "Estadi FIGO",
            "recep_est_porcent": "Receptors Estrogen (%)",
            "rece_de_Ppor": "Receptors Progesterona (%)",
            "edad": "Edat",
            "tto_1_quirugico": "Tractament Quirúrgic",
            "histo_defin": "Tipus Histològic",
            "metasta_distan": "Metàstasi a Distància"
        }
        
        # Només mostrar variables que tenen sentit per PDP
        PDP_FEATURES = list(CATEGORICAL_FEATURES.keys()) + CONTINUOUS_FEATURES
        
        pdp_feature = st.selectbox(
            "Selecciona una variable per veure el PDP:", 
            PDP_FEATURES,
            format_func=lambda x: PDP_FEATURE_NAMES.get(x, x)
        )
        
        if pdp_feature and model_loaded:
            try:
                bg_data_path = BASE_DIR / "data" / "processed" / "preprocessed_v1.csv"
                if bg_data_path.exists():
                    bg_df = pd.read_csv(bg_data_path)
                    X_bg = bg_df[SELECTED_FEATURES]
                    
                    feat_idx = SELECTED_FEATURES.index(pdp_feature)
                    X_mean = X_bg.mean().values
                    
                    fig_pdp, ax_pdp = plt.subplots(figsize=(10, 5))
                    
                    if pdp_feature in CATEGORICAL_FEATURES:
                        # Variable categòrica -> usar barres
                        cat_map = CATEGORICAL_FEATURES[pdp_feature]
                        categories = sorted(cat_map.keys())
                        pdp_values = []
                        
                        for cat_val in categories:
                            X_temp = np.tile(X_mean, (1, 1))
                            X_temp[0, feat_idx] = cat_val
                            X_temp_scaled = scaler.transform(X_temp)
                            pred = model.predict_proba(X_temp_scaled)[0][1]
                            pdp_values.append(pred)
                        
                        # Colors segons probabilitat
                        colors = ['#00c853' if p < 0.3 else '#ffab00' if p < 0.6 else '#ff1744' for p in pdp_values]
                        labels = [cat_map.get(c, str(c)) for c in categories]
                        
                        bars = ax_pdp.bar(range(len(categories)), pdp_values, color=colors, edgecolor='white', linewidth=2)
                        ax_pdp.set_xticks(range(len(categories)))
                        ax_pdp.set_xticklabels(labels, rotation=0, fontsize=10)
                        ax_pdp.set_ylim(0, 1)
                        
                        # Afegir valors sobre les barres
                        for bar, val in zip(bars, pdp_values):
                            ax_pdp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                                       f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                    else:
                        # Variable contínua -> usar línia
                        feat_values = X_bg[pdp_feature].values
                        grid_values = np.linspace(feat_values.min(), feat_values.max(), 50)
                        pdp_values = []
                        
                        for val in grid_values:
                            X_temp = np.tile(X_mean, (1, 1))
                            X_temp[0, feat_idx] = val
                            X_temp_scaled = scaler.transform(X_temp)
                            pred = model.predict_proba(X_temp_scaled)[0][1]
                            pdp_values.append(pred)
                        
                        ax_pdp.plot(grid_values, pdp_values, 'b-', linewidth=2)
                        ax_pdp.fill_between(grid_values, pdp_values, alpha=0.3)
                    
                    ax_pdp.set_xlabel(PDP_FEATURE_NAMES.get(pdp_feature, pdp_feature))
                    ax_pdp.set_ylabel("Probabilitat de Recurrència")
                    ax_pdp.set_title(f"Partial Dependence Plot: {PDP_FEATURE_NAMES.get(pdp_feature, pdp_feature)}")
                    ax_pdp.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    st.pyplot(fig_pdp)
                else:
                    st.warning("No s'han trobat dades de fons per generar el PDP.")
            except Exception as e:
                st.error(f"Error generant PDP: {e}")
        
        # --- CASOS SIMILARS ---
        st.divider()
        st.subheader("Casos Clínics Similars")
        
        try:
            bg_data_path = BASE_DIR / "data" / "processed" / "preprocessed_v1.csv"
            if bg_data_path.exists() and st.session_state.input_data is not None:
                bg_df = pd.read_csv(bg_data_path)
                X_bg = bg_df[SELECTED_FEATURES]
                y_bg = bg_df["recidiva_exitus"]
                X_bg_scaled = scaler.transform(X_bg)
                
                # Calcular distància Euclidiana
                input_vec = st.session_state.input_data.values.flatten()
                distances = np.sqrt(np.sum((X_bg_scaled - input_vec) ** 2, axis=1))
                
                # Trobar els 3 més propers
                top_3_idx = np.argsort(distances)[:3]
                
                # Mapejats per fer les features llegibles
                FEATURE_NAMES = {
                    "grupo_de_riesgo_definitivo": "Grup de Risc",
                    "afectacion_linf": "Afectació Limfàtica (LVSI)",
                    "estadiaje_pre_i": "Estadiatge Pre-quirúrgic",
                    "Tratamiento_sistemico_realizad": "Tractament Sistèmic",
                    "grado_histologi": "Grau Histològic",
                    "infiltracion_mi": "Infiltració Miometrial",
                    "imc": "IMC",
                    "FIGO2023": "Estadi FIGO",
                    "recep_est_porcent": "Receptors Estrogen (%)",
                    "rece_de_Ppor": "Receptors Progesterona (%)",
                    "edad": "Edat",
                    "tto_1_quirugico": "Tractament Quirúrgic",
                    "histo_defin": "Tipus Histològic",
                    "metasta_distan": "Metàstasi a Distància"
                }
                
                RISK_MAP = {1: "Baix", 2: "Intermedi", 3: "Intermedi-Alt", 4: "Alt", 5: "Avançat"}
                GRADO_MAP = {1: "Baix grau (G1-G2)", 2: "Alt grau (G3)"}
                INFIL_MAP = {0: "No", 1: "<50%", 2: ">50%", 3: "Serosa"}
                YESNO_MAP = {0: "No", 1: "Sí"}
                ESTAD_MAP = {0: "I", 1: "II", 2: "III-IV"}
                SIST_MAP = {0: "No", 1: "Parcial", 2: "Completa"}
                FIGO_MAP = {1: "IA1", 2: "IA2", 3: "IA3", 4: "IB", 5: "IC", 6: "IIA", 7: "IIB", 8: "IIC", 9: "IIIA", 10: "IIIB", 11: "IIIC", 12: "IVA", 13: "IVB", 14: "IVC"}
                HISTO_MAP = {1: "Hiperplàsia", 2: "Endometrioide", 3: "Serós", 4: "Cèl·lules clares", 5: "Indiferenciat", 6: "Mixt", 7: "Escamós", 8: "Carcinosarcoma", 9: "Altres"}
                
                def format_value(col, val):
                    try:
                        v = int(round(val))
                    except:
                        v = val
                    if col == "grupo_de_riesgo_definitivo":
                        return RISK_MAP.get(v, str(v))
                    elif col == "grado_histologi":
                        return GRADO_MAP.get(v, str(v))
                    elif col == "infiltracion_mi":
                        return INFIL_MAP.get(v, str(v))
                    elif col in ["afectacion_linf", "tto_1_quirugico", "metasta_distan"]:
                        return YESNO_MAP.get(v, str(v))
                    elif col == "estadiaje_pre_i":
                        return ESTAD_MAP.get(v, str(v))
                    elif col == "Tratamiento_sistemico_realizad":
                        return SIST_MAP.get(v, str(v))
                    elif col == "FIGO2023":
                        return FIGO_MAP.get(v, str(v))
                    elif col == "histo_defin":
                        return HISTO_MAP.get(v, str(v))
                    elif col in ["recep_est_porcent", "rece_de_Ppor", "imc"]:
                        return f"{val:.1f}"
                    elif col == "edad":
                        return str(v)
                    return str(val)
                
                # Obtenir valors originals del nostre cas (desescalats)
                RISK_INV = {"Riesgo bajo": 1, "Riesgo intermedio": 2, "Riesgo intermedio-alto": 3, "Riesgo alto": 4, "Avanzados": 5}
                GRADO_INV = {"Bajo grado (G1-G2)": 1, "Alto grado (G3)": 2}
                INFIL_INV = {"No infiltracion": 0, "Infiltracion miometrial <50%": 1, "Infiltracion miometrial >50%": 2, "Infiltracion serosa": 3}
                LINF_INV = {"No": 0, "Si": 1}
                ESTAD_INV = {"Estadio I": 0, "Estadio II": 1, "Estadio III y IV": 2}
                SIST_INV = {"No realizada": 0, "Dosis parcial": 1, "Dosis completa": 2}
                FIGO_INV = {"IA1": 1, "IA2": 2, "IA3": 3, "IB": 4, "IC": 5, "IIA": 6, "IIB": 7, "IIC": 8, "IIIA": 9, "IIIB": 10, "IIIC": 11, "IVA": 12, "IVB": 13, "IVC": 14}
                QUIR_INV = {"No": 0, "Si": 1}
                HISTO_INV = {"Hiperplasia con atipias": 1, "Carcinoma endometrioide": 2, "Carcinoma seroso": 3, "Carcinoma de celulas claras": 4, "Carcinoma Indiferenciado": 5, "Carcinoma mixto": 6, "Carcinoma escamoso": 7, "Carcinosarcoma": 8, "Otros": 9}
                META_INV = {"No": 0, "Si": 1}
                
                our_case_original = {
                    "grupo_de_riesgo_definitivo": RISK_INV.get(st.session_state.grupo_riesgo, 1),
                    "afectacion_linf": LINF_INV.get(st.session_state.afect_linf, 0),
                    "estadiaje_pre_i": ESTAD_INV.get(st.session_state.estadiaje_pre, 0),
                    "Tratamiento_sistemico_realizad": SIST_INV.get(st.session_state.tto_sistemico, 0),
                    "grado_histologi": GRADO_INV.get(st.session_state.grado, 1),
                    "infiltracion_mi": INFIL_INV.get(st.session_state.infiltracion, 1),
                    "imc": st.session_state.imc if st.session_state.imc else 29.4,
                    "FIGO2023": FIGO_INV.get(st.session_state.figo, 1),
                    "recep_est_porcent": st.session_state.recep_est if st.session_state.recep_est else 90.0,
                    "rece_de_Ppor": st.session_state.recep_prog if st.session_state.recep_prog else 90.0,
                    "edad": st.session_state.edad if st.session_state.edad else 65,
                    "tto_1_quirugico": QUIR_INV.get(st.session_state.tto_quirurgico, 1),
                    "histo_defin": HISTO_INV.get(st.session_state.histo, 2),
                    "metasta_distan": META_INV.get(st.session_state.metasta, 0),
                }
                
                # Construir la taula comparativa
                outcomes = [y_bg.iloc[idx] for idx in top_3_idx]
                
                # Crear DataFrame per a la taula
                table_data = []
                for feat in SELECTED_FEATURES:
                    row = {
                        "Variable": FEATURE_NAMES.get(feat, feat),
                        "Cas Actual": format_value(feat, our_case_original[feat]),
                    }
                    for i, idx in enumerate(top_3_idx):
                        case_data = X_bg.iloc[idx]
                        outcome_emoji = "🟢" if outcomes[i] == 0 else "🔴"
                        row[f"Similar #{i+1} {outcome_emoji}"] = format_value(feat, case_data[feat])
                    table_data.append(row)
                
                # Afegir fila de resultat
                result_row = {
                    "Variable": "**RESULTAT**",
                    "Cas Actual": f"Predicció: {prob:.1%}",
                }
                for i, idx in enumerate(top_3_idx):
                    outcome = outcomes[i]
                    outcome_emoji = "🟢" if outcome == 0 else "🔴"
                    outcome_text = "No Recidiva" if outcome == 0 else "Recidiva"
                    result_row[f"Similar #{i+1} {outcome_emoji}"] = outcome_text
                table_data.insert(0, result_row)
                
                # Generar taula HTML estilitzada
                html_table = """
                <style>
                .comparison-table {
                    width: 100%;
                    border-collapse: separate;
                    border-spacing: 0;
                    font-family: 'Segoe UI', sans-serif;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                }
                .comparison-table th {
                    background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                    color: white;
                    padding: 15px 12px;
                    text-align: center;
                    font-weight: 600;
                    font-size: 14px;
                    border-bottom: 3px solid #4a5568;
                }
                .comparison-table th.current-case {
                    background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%);
                }
                .comparison-table th.similar-green {
                    background: linear-gradient(135deg, #276749 0%, #38a169 100%);
                }
                .comparison-table th.similar-red {
                    background: linear-gradient(135deg, #9b2c2c 0%, #c53030 100%);
                }
                .comparison-table td {
                    padding: 12px;
                    text-align: center;
                    border-bottom: 1px solid rgba(102, 126, 234, 0.2);
                    font-size: 13px;
                }
                .comparison-table tr:nth-child(even) {
                    background-color: rgba(102, 126, 234, 0.05);
                }
                .comparison-table tr:hover {
                    background-color: rgba(102, 126, 234, 0.1);
                    transition: background-color 0.3s ease;
                }
                .comparison-table td.var-name {
                    font-weight: 600;
                    text-align: left;
                    background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, transparent 100%);
                    color: #4a5568;
                }
                .comparison-table td.current-val {
                    background: linear-gradient(90deg, rgba(245, 87, 108, 0.1) 0%, transparent 100%);
                    font-weight: 500;
                    color: #c53030;
                }
                .comparison-table tr.result-row {
                    background: linear-gradient(90deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                    font-weight: 700;
                }
                .comparison-table tr.result-row td {
                    padding: 15px 12px;
                    font-size: 14px;
                    border-top: 2px solid #667eea;
                }
                .result-good { color: #38a169; }
                .result-bad { color: #e53e3e; }
                </style>
                <table class="comparison-table">
                <thead><tr>
                    <th>Variable</th>
                    <th class="current-case">🎯 Cas Actual</th>
                """
                
                # Headers per als casos similars
                for i, idx in enumerate(top_3_idx):
                    outcome = outcomes[i]
                    emoji = "🟢" if outcome == 0 else "🔴"
                    class_name = "similar-green" if outcome == 0 else "similar-red"
                    outcome_text = "No Recidiva" if outcome == 0 else "Recidiva"
                    html_table += f'<th class="{class_name}">{emoji} Similar #{i+1}<br><small>({outcome_text})</small></th>'
                
                html_table += "</tr></thead><tbody>"
                
                # Files de dades
                for feat in SELECTED_FEATURES:
                    var_name = FEATURE_NAMES.get(feat, feat)
                    current_val = format_value(feat, our_case_original[feat])
                    
                    html_table += f'<tr><td class="var-name">{var_name}</td>'
                    html_table += f'<td class="current-val">{current_val}</td>'
                    
                    for i, idx in enumerate(top_3_idx):
                        case_data = X_bg.iloc[idx]
                        val = format_value(feat, case_data[feat])
                        html_table += f'<td>{val}</td>'
                    
                    html_table += '</tr>'
                
                # Fila de resultat
                html_table += f'<tr class="result-row"><td class="var-name">📊 RESULTAT</td>'
                html_table += f'<td class="current-val">Predicció: {prob:.1%}</td>'
                for i in range(3):
                    outcome = outcomes[i]
                    if outcome == 0:
                        html_table += '<td class="result-good">✓ No Recidiva</td>'
                    else:
                        html_table += '<td class="result-bad">✗ Recidiva</td>'
                html_table += '</tr>'
                
                html_table += "</tbody></table>"
                
                st.markdown(html_table, unsafe_allow_html=True)
                
                # Resum
                n_no_recidiva = outcomes.count(0)
                n_recidiva = 3 - n_no_recidiva
                
                st.markdown("---")
                col_summary1, col_summary2 = st.columns(2)
                with col_summary1:
                    st.metric(label="Casos Sense Recidiva", value=f"{n_no_recidiva}/3", delta="Favorable" if n_no_recidiva >= 2 else None)
                with col_summary2:
                    st.metric(label="Casos Amb Recidiva", value=f"{n_recidiva}/3", delta="Desfavorable" if n_recidiva >= 2 else None, delta_color="inverse")
                    
            else:
                st.warning("No s'han trobat dades històriques per comparar.")
        except Exception as e:
            st.error(f"Error trobant casos similars: {e}")


