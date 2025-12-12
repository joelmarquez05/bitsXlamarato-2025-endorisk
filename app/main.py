"""
NEST - Test inicial de l'aplicació Streamlit.
Aquesta és una versió de prova per verificar que Docker funciona.
"""
import streamlit as st

st.set_page_config(
    page_title="NEST - Test",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 NEST - Endometrial Cancer Risk Predictor")
st.success("✅ Docker funciona correctament!")

st.header("Test de Components")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Panel 1: Predicció")
    st.metric(label="Probabilitat de Recaiguda", value="---%", delta="Pendent")
    st.info("El model encara no està entrenat")

with col2:
    st.subheader("📈 Panel 2: Interpretabilitat")
    st.write("Aquí anirà el gràfic d'importància de features")
    st.progress(0.35, text="Grado Tumoral")
    st.progress(0.28, text="Invasión Miometrial")
    st.progress(0.15, text="LVSI")

with col3:
    st.subheader("🤖 Panel 3: Explicació IA")
    st.write("Aquí anirà la resposta de Gemini API")
    st.warning("Gemini API no configurada")

st.divider()

st.header("🧪 Test de Formulari")

with st.form("patient_form"):
    st.subheader("Dades del Pacient")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        age = st.number_input("Edat", min_value=18, max_value=100, value=55)
        tumor_grade = st.selectbox("Grau Tumoral", options=[1, 2, 3])
        stage = st.selectbox("Estadi", options=["I", "II", "III", "IV"])
    
    with col_b:
        myometrial_invasion = st.slider("Invasió Miometrial (%)", 0, 100, 50)
        lvsi = st.checkbox("LVSI Present")
        tumor_size = st.number_input("Mida Tumor (cm)", min_value=0.0, max_value=20.0, value=3.0)
    
    submitted = st.form_submit_button("🔮 Calcular Risc", use_container_width=True)
    
    if submitted:
        st.success(f"""
        **Dades rebudes correctament:**
        - Edat: {age}
        - Grau Tumoral: {tumor_grade}
        - Estadi: {stage}
        - Invasió: {myometrial_invasion}%
        - LVSI: {'Sí' if lvsi else 'No'}
        - Mida: {tumor_size} cm
        
        *El model de predicció encara no està implementat.*
        """)

st.divider()
st.caption("🚀 NEST v0.1 - Test Build | Hackathon Hack the Uterus!")
