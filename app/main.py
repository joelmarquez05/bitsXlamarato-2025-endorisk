"""
NEST - Eina de Predicció de Risc de Càncer d'Endometri.
"""
import streamlit as st

# Configuració de la pàgina
st.set_page_config(
    page_title="NEST - Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 NEST")
st.markdown("**NSMP Endometrial Stratification Tool**")

# Definim les pestanyes
tab1, tab2 = st.tabs(["📝 Dades del Pacient", "📊 Resultats"])

# --- TAB 1: INPUTS ---
with tab1:
    st.header("Introducció de Dades")
    with st.container(border=True):
        with st.form("patient_form_tabs"):
            col1, col2 = st.columns([0.45,0.55], gap="large")
            
            with col1:
                st.subheader("Informació Clínica")
                age = st.number_input("Edat (anys)", min_value=18, max_value=100, value=60)
                stage = st.selectbox("Estadi FIGO", options=["I", "II", "III", "IV"])

            with col2:
                st.subheader("Marcadors Tumorals")
                myometrial_invasion = st.slider("Invasió Miometrial (%)", 0, 100, 0)
                tumor_size = st.number_input("Mida del Tumor (cm)", 0.0, 20.0, 2.0)
                tumor_grade = st.selectbox("Grau Tumoral", options=[1, 2, 3])
                lvsi = st.checkbox("LVSI Present", value=False)
            
            st.markdown("---")
            submitted = st.form_submit_button("🔮 Calcular Risc", use_container_width=True, type="primary")

# --- ESTAT DE SESSIÓ PER A RESULTATS ---
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if submitted:
    st.session_state.prediction_done = True
    # Simulació de càlcul
    import random
    st.session_state.prob = random.uniform(0.1, 0.9)
    st.session_state.risk = "Baix" if st.session_state.prob < 0.3 else "Mitjà" if st.session_state.prob < 0.6 else "Alt"
    st.session_state.color = "#4CAF50" if st.session_state.prob < 0.3 else "#FF9800" if st.session_state.prob < 0.6 else "#F44336"
    
    st.success("✅ Càlcul completat! Ves a la pestanya 'Resultats' per veure l'informe.")

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
            st.markdown(f"<h1 style='text-align: center; color: {st.session_state.color}; font-size: 60px;'>{st.session_state.prob:.1%}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Nivell: <b>{st.session_state.risk}</b></h3>", unsafe_allow_html=True)

        with col_metrics:
            st.subheader("Factors Determinants")
            st.progress(0.8, text="Grau Tumoral (Simulat)")
            st.progress(0.4, text="Invasió Miometrial (Simulat)")

        st.divider()
        st.subheader("🤖 Interpretació Clínica")
        st.info("Aquí apareixeria l'explicació detallada generada per la IA.")
