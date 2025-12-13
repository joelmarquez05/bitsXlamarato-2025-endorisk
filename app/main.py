import pandas as pd
import os
import streamlit as st

# --- CONFIGURACIÓ DE DADES ---
DATA_PATH = "data/raw/IQ_Cancer_Endometrio_merged_NMSP.csv"

# CONSTANTS MAPPING (PLACEHOLDERS - TO BE UPDATED BY USER)
COL_ID = "codigo_participante"
# Features
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

@st.cache_data
def load_data(file_buffer=None):
    # Si no es puja res, fem servir el fitxer local si existeix
    source = file_buffer if file_buffer else DATA_PATH
    
    if not source and not os.path.exists(DATA_PATH):
        return None
        
    try:
        # Determinar tipus de fitxer
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
    except Exception as e:
        # st.error(f"Error carregant dades: {e}") # Ocultem l'error inicial
        return None

# Configuració de la pàgina
st.set_page_config(
    page_title="NEST - Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 NEST")
st.markdown("**NSMP Endometrial Stratification Tool**")

# --- SIDEBAR: SIMULACIÓ BASE DE DADES HOSPITAL ---
with st.sidebar:
    st.header("📂 Base de Dades Hospital")
    
    # File Uploader
    uploaded_file = st.file_uploader("Carregar Llistat Pacients (CSV/XLSX)", type=["csv", "xlsx"])
    
    # Carregar dades (del fitxer pujat o del defecte)
    df = load_data(uploaded_file)
    
    patient_id_input = st.text_input("Buscar ID Pacient", placeholder="Ex: 12345")
    
    patient = None
    
    if df is not None:
        st.info(f"Dades carregades: {len(df)} pacients")
        if patient_id_input:
            row = df[df[COL_ID] == patient_id_input]
            if not row.empty:
                st.success(f"Pacient {patient_id_input} trobat!")
                patient = row.iloc[0]
                
                if st.button("📥 Importar Dades Pacient"):
                    # Helper to set state safely with validation
                    def set_state(key, col_name, cast_type=None, min_val=None, max_val=None, options=None, mapping=None):
                        # Default to None (Blank) if anything goes wrong or data is missing
                        final_val = None
                        
                        if col_name in df.columns:
                            raw_val = patient[col_name]
                            
                            # Check for NA/Null
                            if pd.notna(raw_val):
                                val = raw_val
                                is_valid = True
                                
                                # 1. Casters
                                if cast_type:
                                    try:
                                        val = cast_type(val)
                                    except:
                                        is_valid = False
                                
                                if is_valid:
                                    # 2. Mapping
                                    if mapping:
                                        if val in mapping:
                                            val = mapping[val]
                                        else:
                                            is_valid = False # Mapped value not found
                                    
                                    # 3. Numeric Constraints
                                    if is_valid and (min_val is not None or max_val is not None):
                                        if min_val is not None and val < min_val: val = min_val  # Clamp? Or Invalid? User previously liked Clamp. Let's keep Clamp for numbers.
                                        if max_val is not None and val > max_val: val = max_val

                                    # 4. Option Constraints (Strict)
                                    if is_valid and options is not None:
                                        if val not in options:
                                            is_valid = False
                                    
                                    if is_valid:
                                        final_val = val

                        st.session_state[key] = final_val

                    # --- MAPPING & IMPORT ---
                    # Numeric with constraints
                    set_state("edad", COL_EDAD, int, min_val=18, max_val=120)
                    set_state("imc", COL_IMC, float, min_val=10.0, max_val=60.0)
                    set_state("recep_est", COL_RECEP_EST, float, min_val=0.0, max_val=100.0)
                    set_state("recep_prog", COL_RECEP_PROG, float, min_val=0.0, max_val=100.0)
                    
                    # Mappings defined previously
                    RISK_MAPPING = {
                        1: "Riesgo bajo",
                        2: "Riesgo intermedio",
                        3: "Riesgo intermedio-alto",
                        4: "Riesgo alto",
                        5: "Avanzados"
                    }
                    RISK_OPTIONS = list(RISK_MAPPING.values())
                    set_state("grupo_riesgo", COL_GRUPO_RIESGO_DEFINITIVO, cast_type=int, mapping=RISK_MAPPING, options=RISK_OPTIONS)
                    
                    GRADO_MAPPING = {
                        1: "Bajo grado (G1-G2)",
                        2: "Alto grado (G3)"
                    }
                    GRADO_OPTIONS = list(GRADO_MAPPING.values())
                    set_state("grado", COL_GRADO, cast_type=int, mapping=GRADO_MAPPING, options=GRADO_OPTIONS)

                    INFILTRACION_MAPPING = {
                        0: "No infiltracion",
                        1: "Infiltracion miometrial <50%",
                        2: "Infiltracion miometrial >50%",
                        3: "Infiltracion serosa"
                    }
                    INFILTRACION_OPTIONS = list(INFILTRACION_MAPPING.values())
                    set_state("infiltracion", COL_INFILTRACION, cast_type=int, mapping=INFILTRACION_MAPPING, options=INFILTRACION_OPTIONS)
                    
                    LINF_MAPPING = {0: "No", 1: "Si"}
                    LINF_OPTIONS = ["No", "Si"]
                    set_state("afect_linf", COL_AFECTACION_LINF, cast_type=int, mapping=LINF_MAPPING, options=LINF_OPTIONS)
                    
                    ESTADIAJE_MAPPING = {
                        0: "Estadio I",
                        1: "Estadio II",
                        2: "Estadio III y IV"
                    }
                    ESTADIAJE_OPTIONS = list(ESTADIAJE_MAPPING.values())
                    set_state("estadiaje_pre", COL_ESTADIAJE_PRE, cast_type=int, mapping=ESTADIAJE_MAPPING, options=ESTADIAJE_OPTIONS)
                    
                    SISTEMICO_MAPPING = {
                        0: "No realizada",
                        1: "Dosis parcial",
                        2: "Dosis completa"
                    }
                    SISTEMICO_OPTIONS = list(SISTEMICO_MAPPING.values())
                    set_state("tto_sistemico", COL_TTO_SISTEMICO, cast_type=int, mapping=SISTEMICO_MAPPING, options=SISTEMICO_OPTIONS)
                    
                    FIGO_MAPPING = {
                        1: "IA1", 2: "IA2", 3: "IA3", 4: "IB", 5: "IC",
                        6: "IIA", 7: "IIB", 8: "IIC",
                        9: "IIIA", 10: "IIIB", 11: "IIIC",
                        12: "IVA", 13: "IVB", 14: "IVC"
                    }
                    FIGO_OPTIONS = list(FIGO_MAPPING.values())
                    set_state("figo", COL_FIGO, cast_type=int, mapping=FIGO_MAPPING, options=FIGO_OPTIONS)

                    QUIRURGICO_MAPPING = {0: "No", 1: "Si"}
                    QUIRURGICO_OPTIONS = ["No", "Si"]
                    set_state("tto_quirurgico", COL_TTO_QUIRURGICO, cast_type=int, mapping=QUIRURGICO_MAPPING, options=QUIRURGICO_OPTIONS)
                    
                    HISTO_MAPPING = {
                        1: "Hiperplasia con atipias",
                        2: "Carcinoma endometrioide",
                        3: "Carcinoma seroso",
                        4: "Carcinoma de celulas claras",
                        5: "Carcinoma Indiferenciado",
                        6: "Carcinoma mixto",
                        7: "Carcinoma escamoso",
                        8: "Carcinosarcoma",
                        9: "Otros"
                    }
                    HISTO_OPTIONS = list(HISTO_MAPPING.values())
                    set_state("histo", COL_HISTO, cast_type=int, mapping=HISTO_MAPPING, options=HISTO_OPTIONS)
                    
                    METASTA_MAPPING = {0: "No", 1: "Si"}
                    METASTA_OPTIONS = ["No", "Si"]
                    set_state("metasta", COL_METASTA, cast_type=int, mapping=METASTA_MAPPING, options=METASTA_OPTIONS)
                    
                    st.rerun()
            else:
                st.warning("ID no trobat.")
    else:
        st.warning("Pujar un fitxer o assegurar que 'data/raw/...' existeix.")

# --- RESULTATS STATE ---
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# --- STATE INITIALIZATION ---
def init_state():
    # Define keys to persist with None default
    keys = ["edad", "imc", "grupo_riesgo", "estadiaje_pre", "histo", "grado", 
            "infiltracion", "figo", "metasta", "tto_quirurgico", "tto_sistemico", 
            "afect_linf", "recep_est", "recep_prog"]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None

init_state()

# Definim les pestanyes
tab1, tab2 = st.tabs(["📝 Dades del Pacient", "📊 Resultats"])

# --- TAB 1: INPUTS ---
with tab1:
    st.header("Introducció de Dades Clíniques")
    
    st.info("ℹ️ **Nota:** Els camps que quedin en blanc (per manca de dades o error de format) seran **imputats automàticament** pel sistema durant la predicció. Es recomana omplir-los manualment si la informació està disponible per millorar la precisió.")

    with st.container(border=True):
        with st.form("patient_form_tabs"):
            col1, col2, col3 = st.columns(3, gap="medium")
            
            # Columna 1: Dades Bàsiques i Físiques
            with col1:
                st.subheader("👤 Dades Generals")
                st.number_input("Edat (edad)", min_value=18, max_value=120, value=None, key="edad", placeholder="Edat...")
                st.number_input("IMC (imc)", min_value=10.0, max_value=60.0, value=None, format="%.2f", key="imc", placeholder="IMC...")
                
                risk_opts = ["Riesgo bajo", "Riesgo intermedio", "Riesgo intermedio-alto", "Riesgo alto", "Avanzados"]
                st.selectbox("Grup de Risc Definitiu", risk_opts, key="grupo_riesgo", index=None, placeholder="Seleccionar...")
                
                
                estadiaje_opts = ["Estadio I", "Estadio II", "Estadio III y IV"]
                st.selectbox("Estadiaje Pre-quirúrgic", estadiaje_opts, key="estadiaje_pre", index=None, placeholder="Seleccionar...")

            # Columna 2: Característiques Tumorals
            with col2:
                st.subheader("🔬 Histologia i Tumor")
                
                histo_opts = ["Hiperplasia con atipias", "Carcinoma endometrioide", "Carcinoma seroso", 
                              "Carcinoma de celulas claras", "Carcinoma Indiferenciado", "Carcinoma mixto",
                              "Carcinoma escamoso", "Carcinosarcoma", "Otros"]
                st.selectbox("Tipus Histològic (histo_defin)", histo_opts, key="histo", index=None, placeholder="Seleccionar...")
                
                grado_opts = ["Bajo grado (G1-G2)", "Alto grado (G3)"]
                st.selectbox("Grau Histològic (grado_histologi)", grado_opts, key="grado", index=None, placeholder="Seleccionar...")
                
                infil_opts = ["No infiltracion", "Infiltracion miometrial <50%", "Infiltracion miometrial >50%", "Infiltracion serosa"]
                st.selectbox("Infiltració Miometrial (infiltracion_mi)", infil_opts, key="infiltracion", index=None, placeholder="Seleccionar...")
                
                figo_opts = ["IA1", "IA2", "IA3", "IB", "IC", "IIA", "IIB", "IIC", 
                             "IIIA", "IIIB", "IIIC", "IVA", "IVB", "IVC"]
                st.selectbox("Estadi a 2023", figo_opts, key="figo", index=None, placeholder="Seleccionar...")
                
                st.selectbox("Metàstasi a Distància", ["No", "Si"], key="metasta", index=None, placeholder="Seleccionar...")

            # Columna 3: Tractament i Receptors
            with col3:
                st.subheader("💊 Tractament i Altres")
                st.selectbox("Tractament Quirúrgic 1ari", ["No", "Si"], key="tto_quirurgico", index=None, placeholder="Seleccionar...")
                
                sistemico_opts = ["No realizada", "Dosis parcial", "Dosis completa"]
                st.selectbox("Tractament Sistèmic Realitzat", sistemico_opts, key="tto_sistemico", index=None, placeholder="Seleccionar...")
                st.selectbox("Afectació Limfàtica", ["No", "Si"], key="afect_linf", index=None, placeholder="Seleccionar...")
                st.number_input("Receptors Estrogen (%)", 0.0, 100.0, value=None, key="recep_est", placeholder="0-100%")
                st.number_input("Receptors Progesterona (%)", 0.0, 100.0, value=None, key="recep_prog", placeholder="0-100%")
            
            st.markdown("---")
            submitted = st.form_submit_button("🔮 Calcular Risc de Recurrència", use_container_width=True, type="primary")

    if submitted:
        st.session_state.prediction_done = True
        # Placeholder predicció
        import random
        st.session_state.prob = random.uniform(0.0, 1.0)
        st.success("Càlcul completat. Ves a la pestanya 'Resultats' per veure l'informe.")

# --- TAB 2: RESULTATS ---
with tab2:
    st.header("Resultats de l'Anàlisi")
    
    if not st.session_state.prediction_done:
        st.warning("⚠️ Primer has d'introduir les dades i calcular el risc a la pestanya anterior.")
        st.image("https://illustrations.popsy.co/gray/surr-waiting.svg", width=300)
    else:
        # Mostrar resultats (similars a main.py)
        col_pred, col_metrics = st.columns([1, 2])
        
        with col_pred:
            st.container(border=True)
            st.markdown(f"<h3 style='text-align: center; color: #555555;'>Risc Estimat</h3>", unsafe_allow_html=True)
            # Use a default risk level/color since logic isn't fully here
            risk_level = "Alt" if st.session_state.prob > 0.5 else "Baix"
            color = "#ff4b4b" if st.session_state.prob > 0.5 else "#00c853"
            
            st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 60px;'>{st.session_state.prob:.1%}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Nivell: <b>{risk_level}</b></h3>", unsafe_allow_html=True)

        with col_metrics:
            st.subheader("Factors Determinants")
            st.progress(0.8, text="Grau Tumoral (Simulat)")
            st.progress(0.4, text="Invasió Miometrial (Simulat)")

        st.divider()
        st.subheader("🤖 Interpretació Clínica")
        st.info("Aquí apareixeria l'explicació detallada generada per la IA.")
