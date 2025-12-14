import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import os
import io
from pathlib import Path
from datetime import datetime
import streamlit as st
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- CONFIGURACIÓ DE PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "cancer_endometri.csv"
MODEL_PATH = BASE_DIR / "models" / "svm_model.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
FEATURES_PATH = BASE_DIR / "models" / "selected_features.joblib"

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

def generate_pdf_report(prob, risk_level, original_values, shap_values, features):
    """Genera un informe PDF amb la predicció i dades rellevants."""
    
    # Noms llegibles per les variables
    FEATURE_DISPLAY = {
        "grupo_riesgo": "Grup de Risc Definitiu",
        "afect_linf": "Afectacio Limfatica (LVSI)",
        "estadiaje_pre": "Estadiatge Pre-quirurgic",
        "tto_sistemico": "Tractament Sistemic",
        "grado": "Grau Histologic",
        "infiltracion": "Infiltracio Miometrial",
        "imc": "IMC",
        "figo": "Estadi FIGO 2023",
        "recep_est": "Receptors Estrogen (%)",
        "recep_prog": "Receptors Progesterona (%)",
        "edad": "Edat",
        "tto_quirurgico": "Tractament Quirurgic",
        "histo": "Tipus Histologic",
        "metasta": "Metastasi a Distancia"
    }
    
    def format_original_val(key, val):
        if val is None:
            return "No especificat"
        if isinstance(val, float):
            return f"{val:.1f}"
        return str(val)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Capçalera - Colors corporatius
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(44, 62, 80)  # Gris fosc professional
    pdf.cell(0, 15, "EndoRisk - Informe Clinic", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(127, 140, 141)  # Gris suau
    pdf.cell(0, 8, "Eina de Prediccio de Recurrencia en Cancer Endometrial NSMP", ln=True, align="C")
    pdf.cell(0, 6, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align="C")
    pdf.ln(8)
    
    # Línia separadora
    pdf.set_draw_color(52, 73, 94)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(12)
    
    # Secció de Resultat Principal
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, "RESULTAT DE LA PREDICCIO", ln=True, align="C")
    pdf.ln(3)
    
    # Probabilitat i nivell de risc - Colors professionals
    if prob < 0.30:
        color = (39, 174, 96)  # Verd sobri
        risk_text = "BAIX"
    elif prob < 0.60:
        color = (243, 156, 18)  # Taronja professional
        risk_text = "MODERAT"
    else:
        color = (192, 57, 43)  # Vermell sobri
        risk_text = "ALT"
    
    # Resultat en una línia, sense marc
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, f"{prob:.1%}  -  Risc {risk_text}", align="C", ln=True)
    pdf.ln(5)
    
    pdf.set_text_color(44, 62, 80)
    pdf.set_draw_color(189, 195, 199)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)
    
    # Dades del pacient - Valors originals
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, "DADES DEL PACIENT", ln=True)
    pdf.ln(2)
    
    # Taula de dades
    col_width = 70
    row_height = 6
    
    # Ordre de les variables per mostrar
    display_order = ["edad", "imc", "grupo_riesgo", "estadiaje_pre", "histo", "grado", 
                     "infiltracion", "figo", "metasta", "tto_quirurgico", "tto_sistemico", 
                     "afect_linf", "recep_est", "recep_prog"]
    
    for i, key in enumerate(display_order):
        if i % 2 == 0:
            pdf.set_fill_color(248, 249, 250)
        else:
            pdf.set_fill_color(255, 255, 255)
        
        feat_name = FEATURE_DISPLAY.get(key, key)
        val = original_values.get(key)
        val_str = format_original_val(key, val)
        
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(52, 73, 94)
        pdf.cell(col_width, row_height, feat_name, border=1, fill=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(col_width, row_height, val_str, border=1, fill=True, ln=True)
    
    pdf.ln(4)
    
    # Variables més importants (SHAP) - Bullet points
    if shap_values is not None:
        pdf.set_draw_color(189, 195, 199)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 6, "FACTORS MES IMPORTANTS", ln=True)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, "Variables que mes han influit en la prediccio d'aquest pacient concret:", ln=True)
        pdf.ln(2)
        
        # Mapejat de features internes a noms curts
        SHAP_FEATURE_NAMES = {
            "grupo_de_riesgo_definitivo": "Grup de Risc",
            "afectacion_linf": "LVSI",
            "estadiaje_pre_i": "Estadiatge",
            "Tratamiento_sistemico_realizad": "Tto. Sistemic",
            "grado_histologi": "Grau Histologic",
            "infiltracion_mi": "Infiltracio",
            "imc": "IMC",
            "FIGO2023": "FIGO",
            "recep_est_porcent": "Rec. Estrogen",
            "rece_de_Ppor": "Rec. Progest.",
            "edad": "Edat",
            "tto_1_quirugico": "Tto. Quirurgic",
            "histo_defin": "Histologia",
            "metasta_distan": "Metastasi"
        }
        
        shap_flat = np.array(shap_values).flatten()[:len(features)]
        sorted_idx = np.argsort(np.abs(shap_flat))[::-1][:5]
        
        pdf.set_font("Helvetica", "", 9)
        for idx in sorted_idx:
            feat = features[idx]
            shap_val = shap_flat[idx]
            feat_name = SHAP_FEATURE_NAMES.get(feat, feat)
            effect = "augmenta risc" if shap_val > 0 else "redueix risc"
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 5, f"  - {feat_name}: {effect}", ln=True)
        
        pdf.set_text_color(44, 62, 80)
    
    # Peu de pàgina
    pdf.set_y(-35)
    pdf.set_draw_color(189, 195, 199)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 5, "EndoRisk - Eina de Prediccio de Recurrencia en Cancer Endometrial NSMP", ln=True, align="C")
    pdf.cell(0, 5, "Aquest informe es genera automaticament. No substitueix el criteri medic professional.", ln=True, align="C")
    
    # Retornar bytes
    return bytes(pdf.output())


# Configuració de la pàgina
st.set_page_config(
    page_title="EndoRisk - Predictor",
    page_icon="🏥",
    layout="wide"
)

# --- SPLASH SCREEN (CSS pur, carrega en paral·lel) ---
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = True  # Marcar com mostrat immediatament
    
    # Mostrar splash amb logo
    import base64
    logo_path = BASE_DIR / "images" / "logo.png"
    
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        
        splash_html = f"""
        <style>
            /* Ocultar sidebar i header durant splash */
            .splash-active [data-testid="stSidebar"],
            .splash-active [data-testid="stHeader"],
            .splash-active .stDeployButton {{
                display: none !important;
            }}
            
            .splash-container {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: #ffffff;
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 999999;
                animation: fadeOut 0.8s ease-out 3s forwards;
                pointer-events: none;
            }}
            .splash-logo {{
                max-width: 600px;
                max-height: 450px;
                animation: pulse 1.5s ease-in-out infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); opacity: 1; }}
                50% {{ transform: scale(1.05); opacity: 0.9; }}
            }}
            @keyframes fadeOut {{
                from {{ opacity: 1; }}
                to {{ opacity: 0; visibility: hidden; }}
            }}
        </style>
        <script>
            // Afegir classe per ocultar sidebar
            document.body.classList.add('splash-active');
            // Treure classe després de 4s
            setTimeout(function() {{
                document.body.classList.remove('splash-active');
            }}, 4000);
        </script>
        <div class="splash-container">
            <img src="data:image/png;base64,{logo_data}" class="splash-logo" alt="EndoRisk Logo">
        </div>
        """
        st.markdown(splash_html, unsafe_allow_html=True)

st.title("EndoRisk")
st.markdown("**Eina de Predicció de Recurrència en Càncer Endometrial NSMP**")

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
                    
                    RISK_MAPPING = {1: "Risc baix", 2: "Risc intermedi", 3: "Risc intermedi-alt", 4: "Risc alt", 5: "Avançats"}
                    set_state("grupo_riesgo", COL_GRUPO_RIESGO_DEFINITIVO, cast_type=int, mapping=RISK_MAPPING, options=list(RISK_MAPPING.values()))
                    
                    GRADO_MAPPING = {1: "Grau baix (G1-G2)", 2: "Grau alt (G3)"}
                    set_state("grado", COL_GRADO, cast_type=int, mapping=GRADO_MAPPING, options=list(GRADO_MAPPING.values()))

                    INFILTRACION_MAPPING = {0: "Sense infiltració", 1: "Infiltració miometrial <50%", 2: "Infiltració miometrial >50%", 3: "Infiltració serosa"}
                    set_state("infiltracion", COL_INFILTRACION, cast_type=int, mapping=INFILTRACION_MAPPING, options=list(INFILTRACION_MAPPING.values()))
                    
                    LINF_MAPPING = {0: "No", 1: "Sí"}
                    set_state("afect_linf", COL_AFECTACION_LINF, cast_type=int, mapping=LINF_MAPPING, options=["No", "Sí"])
                    
                    ESTADIAJE_MAPPING = {0: "Estadi I", 1: "Estadi II", 2: "Estadi III i IV"}
                    set_state("estadiaje_pre", COL_ESTADIAJE_PRE, cast_type=int, mapping=ESTADIAJE_MAPPING, options=list(ESTADIAJE_MAPPING.values()))
                    
                    SISTEMICO_MAPPING = {0: "No realitzat", 1: "Dosi parcial", 2: "Dosi completa"}
                    set_state("tto_sistemico", COL_TTO_SISTEMICO, cast_type=int, mapping=SISTEMICO_MAPPING, options=list(SISTEMICO_MAPPING.values()))
                    
                    FIGO_MAPPING = {1: "IA1", 2: "IA2", 3: "IA3", 4: "IB", 5: "IC", 6: "IIA", 7: "IIB", 8: "IIC", 9: "IIIA", 10: "IIIB", 11: "IIIC", 12: "IVA", 13: "IVB", 14: "IVC"}
                    set_state("figo", COL_FIGO, cast_type=int, mapping=FIGO_MAPPING, options=list(FIGO_MAPPING.values()))

                    QUIRURGICO_MAPPING = {0: "No", 1: "Sí"}
                    set_state("tto_quirurgico", COL_TTO_QUIRURGICO, cast_type=int, mapping=QUIRURGICO_MAPPING, options=["No", "Sí"])
                    
                    HISTO_MAPPING = {1: "Hiperplàsia amb atípies", 2: "Carcinoma endometrioide", 3: "Carcinoma serós", 4: "Carcinoma de cèl·lules clares", 5: "Carcinoma indiferenciat", 6: "Carcinoma mixt", 7: "Carcinoma escamós", 8: "Carcinosarcoma", 9: "Altres"}
                    set_state("histo", COL_HISTO, cast_type=int, mapping=HISTO_MAPPING, options=list(HISTO_MAPPING.values()))
                    
                    METASTA_MAPPING = {0: "No", 1: "Sí"}
                    set_state("metasta", COL_METASTA, cast_type=int, mapping=METASTA_MAPPING, options=["No", "Sí"])
                    
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
if "n_nan" not in st.session_state:
    st.session_state.n_nan = 0
if "confidence" not in st.session_state:
    st.session_state.confidence = None

def init_state():
    keys = ["edad", "imc", "grupo_riesgo", "estadiaje_pre", "histo", "grado", 
            "infiltracion", "figo", "metasta", "tto_quirurgico", "tto_sistemico", 
            "afect_linf", "recep_est", "recep_prog"]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None

init_state()

# --- PESTANYES ---
tab1, tab2, tab3 = st.tabs(["Dades del Pacient", "Resultats", "Anàlisi Avançada (PCA)"])

# --- TAB 1: INPUTS ---
with tab1:
    st.header("Introducció de Dades Clíniques")
    st.info("**Nota:** Els camps que quedin en blanc seran **imputats automàticament** pel sistema durant la predicció.")

    with st.container(border=True):
        with st.form("patient_form_tabs"):
            col1, col2, col3 = st.columns(3, gap="medium")
            
            with col1:
                st.subheader("Dades Generals")
                st.number_input("Edat (edad)", min_value=18, max_value=120, 
                               value=None, key="edad", placeholder="Edat...")
                st.number_input("IMC (imc)", min_value=10.0, max_value=60.0, 
                               value=None, format="%.2f", key="imc", placeholder="IMC...")
                risk_opts = ["Risc baix", "Risc intermedi", "Risc intermedi-alt", "Risc alt", "Avançats"]
                st.selectbox("Grup de Risc Definitiu", risk_opts, key="grupo_riesgo", 
                            index=None, placeholder="Seleccionar...")
                estadiaje_opts = ["Estadi I", "Estadi II", "Estadi III i IV"]
                st.selectbox("Estadiatge Pre-quirúrgic", estadiaje_opts, key="estadiaje_pre", 
                            index=None, placeholder="Seleccionar...")

            with col2:
                st.subheader("Histologia i Tumor")
                histo_opts = ["Hiperplàsia amb atípies", "Carcinoma endometrioide", "Carcinoma serós", "Carcinoma de cèl·lules clares", "Carcinoma indiferenciat", "Carcinoma mixt", "Carcinoma escamós", "Carcinosarcoma", "Altres"]
                st.selectbox("Tipus Histològic", histo_opts, key="histo", 
                            index=None, placeholder="Seleccionar...")
                grado_opts = ["Grau baix (G1-G2)", "Grau alt (G3)"]
                st.selectbox("Grau Histològic", grado_opts, key="grado", 
                            index=None, placeholder="Seleccionar...")
                infil_opts = ["Sense infiltració", "Infiltració miometrial <50%", "Infiltració miometrial >50%", "Infiltració serosa"]
                st.selectbox("Infiltració Miometrial", infil_opts, key="infiltracion", 
                            index=None, placeholder="Seleccionar...")
                figo_opts = ["IA1", "IA2", "IA3", "IB", "IC", "IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA", "IVB", "IVC"]
                st.selectbox("Estadi FIGO 2023", figo_opts, key="figo", 
                            index=None, placeholder="Seleccionar...")
                yesno_opts = ["No", "Sí"]
                st.selectbox("Metàstasi a Distància", yesno_opts, key="metasta", 
                            index=None, placeholder="Seleccionar...")

            with col3:
                st.subheader("Tractament i Altres")
                st.selectbox("Tractament Quirúrgic 1ari", yesno_opts, key="tto_quirurgico", 
                            index=None, placeholder="Seleccionar...")
                sistemico_opts = ["No realitzat", "Dosi parcial", "Dosi completa"]
                st.selectbox("Tractament Sistèmic Realitzat", sistemico_opts, key="tto_sistemico", 
                            index=None, placeholder="Seleccionar...")
                st.selectbox("Afectació Limfàtica (LVSI)", yesno_opts, key="afect_linf", 
                            index=None, placeholder="Seleccionar...")
                st.number_input("Receptors Estrogen (%)", 0.0, 100.0, 
                               value=None, key="recep_est", placeholder="0-100%")
                st.number_input("Receptors Progesterona (%)", 0.0, 100.0, 
                               value=None, key="recep_prog", placeholder="0-100%")
            
            st.markdown("---")
            submitted = st.form_submit_button("Calcular Risc de Recurrència", use_container_width=True, type="primary")

    if submitted and model_loaded:
        # Mapejats inversos per convertir UI -> valors numèrics
        RISK_INV = {"Risc baix": 1, "Risc intermedi": 2, "Risc intermedi-alt": 3, "Risc alt": 4, "Avançats": 5}
        GRADO_INV = {"Grau baix (G1-G2)": 1, "Grau alt (G3)": 2}
        INFIL_INV = {"Sense infiltració": 0, "Infiltració miometrial <50%": 1, "Infiltració miometrial >50%": 2, "Infiltració serosa": 3}
        LINF_INV = {"No": 0, "Sí": 1}
        ESTAD_INV = {"Estadi I": 0, "Estadi II": 1, "Estadi III i IV": 2}
        SIST_INV = {"No realitzat": 0, "Dosi parcial": 1, "Dosi completa": 2}
        FIGO_INV = {"IA1": 1, "IA2": 2, "IA3": 3, "IB": 4, "IC": 5, "IIA": 6, "IIB": 7, "IIC": 8, "IIIA": 9, "IIIB": 10, "IIIC": 11, "IVA": 12, "IVB": 13, "IVC": 14}
        QUIR_INV = {"No": 0, "Sí": 1}
        HISTO_INV = {"Hiperplàsia amb atípies": 1, "Carcinoma endometrioide": 2, "Carcinoma serós": 3, "Carcinoma de cèl·lules clares": 4, "Carcinoma indiferenciat": 5, "Carcinoma mixt": 6, "Carcinoma escamós": 7, "Carcinosarcoma": 8, "Altres": 9}
        META_INV = {"No": 0, "Sí": 1}
        
        # =================================================================
        # COMPTAR CAMPS BUITS (NANs) per calcular confiança
        # =================================================================
        n_nan = 0
        input_fields = [
            st.session_state.edad,
            st.session_state.imc,
            st.session_state.grupo_riesgo,
            st.session_state.estadiaje_pre,
            st.session_state.histo,
            st.session_state.grado,
            st.session_state.infiltracion,
            st.session_state.figo,
            st.session_state.metasta,
            st.session_state.tto_quirurgico,
            st.session_state.tto_sistemico,
            st.session_state.afect_linf,
            st.session_state.recep_est,
            st.session_state.recep_prog
        ]
        n_nan = sum(1 for v in input_fields if v is None)
        st.session_state.n_nan = n_nan
        
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
        
        # =================================================================
        # CALCULAR CONFIANÇA: Combina distància frontera SVM + penalització NANs
        # =================================================================
        # Confiança base: distància a la frontera de decisió (0.5)
        # Si prob=0.5 -> confiança_base=0, si prob=0 o 1 -> confiança_base=1
        decision_distance = abs(prob - 0.5) * 2  # Escala de 0 a 1
        
        # Penalització per NANs imputats (màxim 14 camps)
        # Cada NAN redueix la confiança en un 5%
        nan_penalty = n_nan * 0.05
        
        # Confiança final (mínim 0)
        confidence = max(0, decision_distance - nan_penalty)
        st.session_state.confidence = confidence
        
        # SHAP amb dades de fons (optimitzat per velocitat)
        with st.spinner("Calculant interpretabilitat..."):
            try:
                bg_data_path = BASE_DIR / "data" / "processed" / "preprocessed.csv"
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
        
        # --- INDICADOR DE CONFIANÇA ---
        st.divider()
        st.markdown("### Indicador de confiança")
        
        confidence = st.session_state.confidence
        n_nan = st.session_state.n_nan
        
        if confidence is not None:
            col_conf1, col_conf2 = st.columns([2, 1])
            
            with col_conf1:
                # Barra de confiança visual
                if confidence >= 0.7:
                    conf_color = "#00c853"  # Verd
                    conf_text = "Alta"
                    conf_emoji = "✅"
                elif confidence >= 0.4:
                    conf_color = "#ffab00"  # Taronja
                    conf_text = "Moderada"
                    conf_emoji = "⚠️"
                else:
                    conf_color = "#ff1744"  # Vermell
                    conf_text = "Baixa"
                    conf_emoji = "❗"
                
                st.markdown(f"""
                <div style="background: #f0f2f6; border-radius: 10px; padding: 15px; margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: #333;"> Confiança de la Predicció</span>
                        <span style="font-weight: 700; color: {conf_color}; font-size: 1.2em;">{confidence:.0%} ({conf_text})</span>
                    </div>
                    <div style="background: #ddd; border-radius: 5px; height: 12px; overflow: hidden;">
                        <div style="background: {conf_color}; height: 100%; width: {confidence*100}%; transition: width 0.5s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_conf2:
                # Mostrar nombre de camps imputats
                if n_nan == 0:
                    nan_emoji = "✨"
                    nan_msg = "Tots els camps informats"
                elif n_nan <= 3:
                    nan_emoji = "📝"
                    nan_msg = f"{n_nan} camp(s) imputat(s)"
                else:
                    nan_emoji = "⚠️"
                    nan_msg = f"{n_nan} camps imputats"
                
                st.markdown(f"""
                <div style="background: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center; height: 100%;">
                    <div style="font-size: 2em;">{nan_emoji}</div>
                    <div style="font-weight: 600; color: #555; margin-top: 5px;">{nan_msg}</div>
                    <div style="font-size: 0.85em; color: #888;">de 14 variables</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Explicació de la confiança
            with st.expander("ℹ️ Com es calcula la confiança?"):
                st.markdown("""
                L'indicador de confiança combina dos factors:
                
                1. **Certesa del Model (SVM)**: Quan més lluny estigui la probabilitat del 50% (frontera de decisió), més segur està el model de la seva predicció.
                   - Predicció al 95% o 5% → Alta certesa
                   - Predicció al 55% o 45% → Baixa certesa
                
                2. **Completitud de les Dades**: Cada camp que no s'ha informat i ha estat imputat automàticament redueix la confiança.
                
                **Interpretació:**
                - 🟢 **Alta (>70%)**: Predicció fiable, dades completes
                - 🟡 **Moderada (40-70%)**: Considerar amb cautela
                - 🔴 **Baixa (<40%)**: Revisar dades i completar camps buits
                """)
        
        st.divider()
        
        # --- INTERPRETABILITAT ---
        st.header("Interpretabilitat")
        
        # --- SHAP ---
        st.subheader("SHAP - Contribució de cada variable")
        st.info("""
        Cada barra mostra com una variable ha influït en la predicció d'aquest pacient concret.
        - **Barres vermelles**: Aquesta variable ha **augmentat** el risc de recurrència.
        - **Barres verdes**: Aquesta variable ha **reduït** el risc de recurrència.
        - La **longitud** de la barra indica la magnitud de l'impacte.
        """)
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
        st.subheader("PDP - Efecte general d'una variable")
        st.info("""
        El PDP mostra com canviaria la predicció del model **en general** si variés una variable, mantenint les altres constants.
        - Per **variables categòriques**: Cada barra mostra la probabilitat mitjana per a cada categoria.
        - Per **variables contínues**: La línia mostra com varia la probabilitat a mesura que augmenta el valor.
        """)
        
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
                bg_data_path = BASE_DIR / "data" / "processed" / "preprocessed.csv"
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
        st.subheader("Casos clínics similars")
        st.info("""
        Mostrem els 2 pacients històrics amb característiques més semblants al cas actual.
        Això permet comparar el perfil clínic i observar quins van tenir recidiva i quins no.
        """)
        
        try:
            bg_data_path = BASE_DIR / "data" / "processed" / "preprocessed.csv"
            if bg_data_path.exists() and st.session_state.input_data is not None:
                bg_df = pd.read_csv(bg_data_path)
                X_bg = bg_df[SELECTED_FEATURES]
                y_bg = bg_df["recidiva_exitus"]
                X_bg_scaled = scaler.transform(X_bg)
                
                # Calcular distància Euclidiana
                input_vec = st.session_state.input_data.values.flatten()
                distances = np.sqrt(np.sum((X_bg_scaled - input_vec) ** 2, axis=1))
                
                # Ordenar per distància i filtrar els que tenen distància 0 (és el mateix cas)
                sorted_idx = np.argsort(distances)
                similar_idx = [idx for idx in sorted_idx if distances[idx] > 0.001][:2]  # Agafar 2 casos
                
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
                outcomes = [y_bg.iloc[idx] for idx in similar_idx]
                
                # Crear DataFrame per a la taula
                table_data = []
                for feat in SELECTED_FEATURES:
                    row = {
                        "Variable": FEATURE_NAMES.get(feat, feat),
                        "Cas Actual": format_value(feat, our_case_original[feat]),
                    }
                    for i, idx in enumerate(similar_idx):
                        case_data = X_bg.iloc[idx]
                        outcome_emoji = "🟢" if outcomes[i] == 0 else "🔴"
                        row[f"Similar #{i+1} {outcome_emoji}"] = format_value(feat, case_data[feat])
                    table_data.append(row)
                
                # Afegir fila de resultat
                result_row = {
                    "Variable": "**RESULTAT**",
                    "Cas Actual": f"Predicció: {prob:.1%}",
                }
                for i, idx in enumerate(similar_idx):
                    outcome = outcomes[i]
                    outcome_emoji = "🟢" if outcome == 0 else "🔴"
                    outcome_text = "No Recidiva" if outcome == 0 else "Recidiva"
                    result_row[f"Similar #{i+1} {outcome_emoji}"] = outcome_text
                table_data.insert(0, result_row)
                
                # Generar taula HTML estilitzada
                # Determinar color segons probabilitat
                if prob < 0.5:
                    current_header_bg = "linear-gradient(135deg, #276749 0%, #38a169 100%)"
                    current_val_bg = "linear-gradient(90deg, rgba(56, 161, 105, 0.15) 0%, transparent 100%)"
                    current_val_color = "#276749"
                else:
                    current_header_bg = "linear-gradient(135deg, #9b2c2c 0%, #c53030 100%)"
                    current_val_bg = "linear-gradient(90deg, rgba(197, 48, 48, 0.15) 0%, transparent 100%)"
                    current_val_color = "#9b2c2c"
                
                html_table = f"""
                <style>
                .comparison-table {{
                    width: 100%;
                    border-collapse: separate;
                    border-spacing: 0;
                    font-family: 'Segoe UI', sans-serif;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                }}
                .comparison-table th {{
                    background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                    color: white;
                    padding: 15px 12px;
                    text-align: center;
                    font-weight: 600;
                    font-size: 14px;
                    border-bottom: 3px solid #4a5568;
                }}
                .comparison-table th.current-case {{
                    background: {current_header_bg};
                }}
                .comparison-table th.similar-green {{
                    background: linear-gradient(135deg, #276749 0%, #38a169 100%);
                }}
                .comparison-table th.similar-red {{
                    background: linear-gradient(135deg, #9b2c2c 0%, #c53030 100%);
                }}
                .comparison-table td {{
                    padding: 12px;
                    text-align: center;
                    border-bottom: 1px solid rgba(102, 126, 234, 0.2);
                    font-size: 13px;
                }}
                .comparison-table tr:nth-child(even) {{
                    background-color: rgba(102, 126, 234, 0.05);
                }}
                .comparison-table tr:hover {{
                    background-color: rgba(102, 126, 234, 0.1);
                    transition: background-color 0.3s ease;
                }}
                .comparison-table td.var-name {{
                    font-weight: 600;
                    text-align: left;
                    background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, transparent 100%);
                    color: #4a5568;
                }}
                .comparison-table td.current-val {{
                    background: {current_val_bg};
                    font-weight: 500;
                    color: {current_val_color};
                }}
                .comparison-table tr.result-row {{
                    background: linear-gradient(90deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                    font-weight: 700;
                }}
                .comparison-table tr.result-row td {{
                    padding: 15px 12px;
                    font-size: 14px;
                    border-top: 2px solid #667eea;
                }}
                .result-good {{ color: #38a169; }}
                .result-bad {{ color: #e53e3e; }}
                </style>
                <table class="comparison-table">
                <thead><tr>
                    <th>Variable</th>
                    <th class="current-case">Cas Actual</th>
                """
                
                for i, idx in enumerate(similar_idx):
                    outcome = outcomes[i]
                    class_name = "similar-green" if outcome == 0 else "similar-red"
                    outcome_text = "No Recidiva" if outcome == 0 else "Recidiva"
                    html_table += f'<th class="{class_name}">Similar #{i+1}<br><small>({outcome_text})</small></th>'
                
                html_table += "</tr></thead><tbody>"
                
                # Files de dades
                for feat in SELECTED_FEATURES:
                    var_name = FEATURE_NAMES.get(feat, feat)
                    current_val = format_value(feat, our_case_original[feat])
                    
                    html_table += f'<tr><td class="var-name">{var_name}</td>'
                    html_table += f'<td class="current-val">{current_val}</td>'
                    
                    for i, idx in enumerate(similar_idx):
                        case_data = X_bg.iloc[idx]
                        val = format_value(feat, case_data[feat])
                        html_table += f'<td>{val}</td>'
                    
                    html_table += '</tr>'
                
                # Fila de resultat
                html_table += f'<tr class="result-row"><td class="var-name">RESULTAT</td>'
                html_table += f'<td class="current-val">Predicció: {prob:.1%}</td>'
                for i in range(len(similar_idx)):
                    outcome = outcomes[i]
                    if outcome == 0:
                        html_table += '<td class="result-good">No Recidiva</td>'
                    else:
                        html_table += '<td class="result-bad">Recidiva</td>'
                html_table += '</tr>'
                
                html_table += "</tbody></table>"
                
                st.markdown(html_table, unsafe_allow_html=True)
                    
            else:
                st.warning("No s'han trobat dades històriques per comparar.")
        except Exception as e:
            st.error(f"Error trobant casos similars: {e}")
        
        # --- EXPORTAR PDF ---
        st.divider()
        st.subheader("📄 Exportar Informe")
        
        if st.session_state.input_data is not None:
            try:
                # Recollir valors originals del formulari
                original_values = {
                    "edad": st.session_state.get("edad"),
                    "imc": st.session_state.get("imc"),
                    "grupo_riesgo": st.session_state.get("grupo_riesgo"),
                    "estadiaje_pre": st.session_state.get("estadiaje_pre"),
                    "histo": st.session_state.get("histo"),
                    "grado": st.session_state.get("grado"),
                    "infiltracion": st.session_state.get("infiltracion"),
                    "figo": st.session_state.get("figo"),
                    "metasta": st.session_state.get("metasta"),
                    "tto_quirurgico": st.session_state.get("tto_quirurgico"),
                    "tto_sistemico": st.session_state.get("tto_sistemico"),
                    "afect_linf": st.session_state.get("afect_linf"),
                    "recep_est": st.session_state.get("recep_est"),
                    "recep_prog": st.session_state.get("recep_prog"),
                }
                
                pdf_bytes = generate_pdf_report(
                    prob=st.session_state.prob,
                    risk_level=risk_level,
                    original_values=original_values,
                    shap_values=st.session_state.shap_values,
                    features=SELECTED_FEATURES
                )
                
                st.download_button(
                    label="⬇️ Descarregar Informe PDF",
                    data=pdf_bytes,
                    file_name=f"NEST_informe_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
                st.caption("L'informe inclou: predicció, dades del pacient i variables més importants.")
            except Exception as e:
                st.error(f"Error generant PDF: {e}")

# --- TAB 3: ANÀLISI AVANÇADA (PCA) ---
with tab3:
    st.header("Anàlisi Avançada i Segmentació (PCA)")
    
    st.markdown("S'han buscat diferents conjunts de variables per agrupar individus en funció de la seva similitud. És a dir, si dues persones tenen valors similars en la majoria de les variables s'han afegit al mateix grup. Hem buscat separar tots els individus del tipus NSMP en dos grups en funció del risc de recaiguda.")

    # 1. Barres Apilades i Agrupades
    st.subheader("1. Distribució de Risc per Clúster")
    col1, col2 = st.columns(2)
    
    IMAGES_DIR = BASE_DIR / "images"
    
    with col1:
        stack_bar_path = IMAGES_DIR / "stack_bar.png"
        if stack_bar_path.exists():
            st.image(str(stack_bar_path), width='stretch')
        else:
            st.warning("⏳ Imatge `stack_bar.png` pendent de pujar")
    with col2:
        group_bar_path = IMAGES_DIR / "group_bar.png"
        if group_bar_path.exists():
            st.image(str(group_bar_path), width='stretch')
        else:
            st.warning("⏳ Imatge `group_bar.png` pendent de pujar")
    
    st.markdown("A l'esquerra es pot veure la proporció d'individus que han tingut una recaiguda en cada grup. Com es pot veure, hem aconseguit separar individus que poden ser considerats de baix risc, ja que tan sols el 5% d'ells patirà una recaiguda, dels que es poden considerar d'alt risc, on la proporció dels que recauen és de més del 65%.")
    st.markdown("A la dreta es pot veure la quantitat d'individus que han recaigut o no en cada grup. Com es pot observar, més de 100 individus no recauen en el grup de baix risc, mentre que tan sols 6 sís. Pel que fa al grup d'alt risc gairebé 30 dones han tingut recaiguda, mentre que 15 no.")
    
    st.divider()

    # 2. K-Means vs Original
    st.subheader("2. Validació: K-Means vs Realitat Clínica")
    kmeans_path = IMAGES_DIR / "k_means_vs_og.png"
    if kmeans_path.exists():
        st.image(str(kmeans_path), width='stretch')
    else:
        st.warning("⏳ Imatge `k_means_vs_og.png` pendent de pujar")
    
    st.markdown("Hem aplicat PCA per representar els grups en dues dimensions per veure com es diferencien. A l'esquerra hi ha la nostra agrupació, mentre que a la dreta hi ha els grups als quals pertanyen en realitat, en funció de si hi ha hagut recaiguda o no. Clarament és molt similar.")
    st.markdown("Per fer aquest clustering hem fet servir variables ja estudiades i que se sap com influeixen sobre el risc de la pacient. Tot i així, utilitzant-les totes s'ha pogut trobar una clara diferenciació entre dos subgrups dins dels individus del tipus NSMP, on unes tenen risc molt més elevat que les altres i probablement s'hauran de tractar de forma més agressiva o menys.")
    
    st.divider()

    # 3. PCA Comparativa
    st.subheader("3. Cerca de Variables menys Populars (PCA)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Variables Originals**")
        biplot_path = IMAGES_DIR / "biplot.png"
        if biplot_path.exists():
            st.image(str(biplot_path), width='stretch')
        else:
            st.warning("⏳ Imatge `biplot.png` pendent de pujar")
    with col2:
        st.markdown("**Variables Seleccionades**")
        pca_nuevo_path = IMAGES_DIR / "pca_nuevo.png"
        if pca_nuevo_path.exists():
            st.image(str(pca_nuevo_path), width='stretch')
        else:
            st.warning("⏳ Imatge `pca_nuevo.png` pendent de pujar")
    
    st.markdown("En aquests gràfics es pot veure una anàlisi de l’explicabilitat d’algunes variables molt útils per a la separació de pacients en funció del seu risc. A l’esquerra s’han utilitzat les variables més típicament usades, mentre que a la dreta s’han emprat aquelles que no s’utilitzen tan sovint per veure si aporten informació per fer aquesta classificació. Tal com es pot veure, sí que és el cas, i es pot obtenir una separació molt similar. Es pot veure que n_total_GC , abordajeqx i histe_avanz més gran fan que el risc de la pacient sigui considerat com a menor, mentre que si el valor de AP_ganPelv, n_gangP_afec, tx_anexial i AP_glanPaor és alt, el risc probablement també ho serà. De les totes variables no tan utilitzades, les que apareixen en aquest gràfic de la dreta han resultat les més útils per classificar les pacients en funció del seu risc.")
