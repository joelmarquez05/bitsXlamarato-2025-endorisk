"""
NEST - Eina de Predicció de Risc de Càncer d'Endometri.
Versió Vertical (Single Page).
"""
import streamlit as st

# Configuració de la pàgina
st.set_page_config(
    page_title="NEST - Predictor",
    page_icon="🏥",
    layout="wide"
)

# --- CAPÇALERA ---
st.title("🏥 NEST")
st.markdown("**NSMP Endometrial Stratification Tool**")
st.markdown("---")

# --- AREA D'ENTRADA DE DADES (VERTICAL) ---
st.header("📋 Dades de la Pacient")
st.info("Introdueix les dades clíniques a continuació per calcular el risc de recaiguda.")

with st.container(border=True):
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informació Clínica")
            age = st.number_input("Edat (anys)", min_value=18, max_value=100, value=60)
            
            tumor_grade = st.selectbox(
                "Grau Tumoral", 
                options=[1, 2, 3],
                help="1=Ben diferenciat, 3=Pobrement diferenciat"
            )
            
            stage = st.selectbox(
                "Estadi FIGO", 
                options=["I", "II", "III", "IV"]
            )

        with col2:
            st.subheader("Marcadors Tumorals")
            myometrial_invasion = st.slider(
                "Invasió Miometrial (%)", 
                min_value=0, 
                max_value=100, 
                value=0
            )
            
            tumor_size = st.number_input(
                "Mida del Tumor (cm)", 
                min_value=0.0, 
                max_value=20.0, 
                value=2.0,
                step=0.1
            )
            
            lvsi = st.checkbox("LVSI Present", value=False)
        
        st.markdown("---")
        submitted = st.form_submit_button("🔮 Calcular Risc de Recaiguda", use_container_width=True, type="primary")

# --- RESULTATS ---

if submitted:
    # --- LOGICA SIMULADA ---
    import random
    mock_prob = random.uniform(0.1, 0.9)
    risk_level = "Baix" if mock_prob < 0.3 else "Mitjà" if mock_prob < 0.6 else "Alt"
    risk_color = "green" if mock_prob < 0.3 else "orange" if mock_prob < 0.6 else "red"
    
    st.markdown("---")
    st.header("📊 Resultats de l'Anàlisi")
    
    # 1. PANELL PRINCIPAL
    col_pred, col_metrics = st.columns([1, 2])
    
    with col_pred:
        st.container(border=True)
        st.markdown(f"<h3 style='text-align: center; color: gray;'>Risc Estimat</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: {risk_color}; font-size: 60px;'>{mock_prob:.1%}</h1>", unsafe_allow_html=True)
        
        risk_html = f"""
        <div style='text-align: center; padding: 10px; background-color: {risk_color}; color: white; border-radius: 5px;'>
            <h3>Nivell: {risk_level}</h3>
        </div>
        """
        st.markdown(risk_html, unsafe_allow_html=True)

    with col_metrics:
        st.subheader("Factors Determinants")
        st.progress(0.8 if tumor_grade == 3 else 0.4, text="Grau Tumoral")
        st.progress(myometrial_invasion / 100, text="Invasió Miometrial")
        st.progress(0.9 if lvsi else 0.1, text="LVSI")

    # 2. EXPLICACIÓ IA
    st.markdown("### 🤖 Interpretació Clínica (IA)")
    with st.container(border=True):
        st.info(f"Anàlisi generada per al pacient de {age} anys, estadi {stage}.")
        st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

