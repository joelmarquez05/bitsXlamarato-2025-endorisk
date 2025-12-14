# EndoRisk: NSMP Endometrial Risk Stratification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED)

> **Hack the Uterus! - NEST Challenge**
> *Bridging the gap between molecular biology and clinical decision-making.*

## Inspiració
La nostra inspiració neix de la combinació de medicina i la ciència de dades. Com a equip, ja teníem experiència prèvia a l'haver participat en un projecte relacionat amb la ginecologia, fet que va despertar immediatament el nostre interès per aquest repte en particular. Crèiem que la creació d'un model de predicció sumada a una anàlisi profunda i rigorosa de la base de dades clínica, s'adequava perfectament a les nostres habilitats i interessos com a científics de dades. Aportar una solució tangible que pogués ajudar els oncòlegs en la presa de decisions difícils en un grup de pacients amb una gran complexitat de classificació era el nostre objectiu principal.

## Què Fa Endorisk
**Endorisk** és un Sistema de Suport a la Decisió Clínica (CDSS) impulsat per *Machine Learning*, dissenyat específicament per al subgrup molecular NSMP (Non-Specific Molecular Profile) del càncer d'endometri. La seva funció principal és classificar els pacients segons el seu risc (Baix o Alt) utilitzant variables clíniques i biomarcadors.

Prioritzem un enfocament de "**Safety-First AI**":
1.  **Estratificació de Risc**: Classifica els pacients en Alt o Baix Risc utilitzant variables clíniques clau (com invasió miometrial o grau histològic) i biomarcadors.
2.  **Informació Accionable**: Proporciona una sortida clara per a la interpretació clínica immediata a través d'una interfície intuïtiva.
3.  **Explicabilitat**: Utilitza **SHAP** per mostrar *per què* s'ha assignat una puntuació de risc específica, generant confiança amb els professionals mèdics i evitant l'efecte "caixa negra".

## Com ho hem creat
Hem construït Endorisk seguint una arquitectura modular i robusta, reflectida en l'estructura del nostre repositori:

1.  **Flux de Data Science (Notebooks & Models)**:
    *   El nucli del projecte es basa en un pipeline seqüencial de 4 notebooks (Exploració, Preprocessament, Entrenament i Avaluació).
    *   Hem utilitzat **Python** amb **Pandas** i **NumPy** per a la manipulació de dades.
    *   Per al modelatge, hem combinat **Scikit-learn** amb la potència de **XGBoost** per explorar totes les possibiltats de modelatge que complissin les necessitats tècniques i de interpretabilitat del repte.
    *   El model final escollit es una Support Vector Machine (SVM) per a la classificació binaria de les prediccions en funció de si els pacients tindran recaiguda o mort i, per tant, són de alt risc.
    *   Hem integrat **SHAP** per a l'explicabilitat matemàtica (XAI), permetent visualitzar l'impacte exacte de cada variable en la predicció.
    *   La persistència dels models i escaladors es gestiona amb **Joblib**, assegurant que el model que corre a l'app és idèntic a l'entrenat.

2.  **Frontend (Streamlit App)**:
    *   La interfície web s'ha desenvolupat amb **Streamlit**, estructurada en fitxers modulars dins la carpeta `app/`.
    *   Hem dissenyat la UX utilitzant `st.tabs` per separar netament l'entrada de dades ("Dades del Pacient") de la visualització de prediccions ("Resultats"). Un tercer tab ("Anàlisi Avançada") permet visualitzar les conclusions que hem obtingut al estudiar la base de dades, que aporten informació sobre les variables predictores.
    *   Fem un ús extensiu de `st.session_state` per mantenir la persistència de les dades del pacient durant la navegació .
    *   Hem implementat una funcionalitat de simulació a la `st.sidebar` que permet carregar fitxers Excel/CSV (`openpyxl`) simulant una connexió amb la base de dades de l'hospital, amb una lògica robusta de càrrega de dades (`load_data`) que gestiona diferents codificacions.

3.  **DevOps e Infraestructura**:
    *   Tot el projecte està containeritzat utilitzant **Docker** i orquestrat amb `docker-compose`, garantint la reproductibilitat de l'entorn i facilitant el desplegament "plug-and-play" en qualsevol màquina, evitant errors de compatibilitat.

## Reptes que hem trobat
El repte tècnic més significatiu ha estat la **alta dimensionalitat de la base de dades original**, on ens hem trobat amb una situació crítica: el nombre de columnes (variables) superava al nombre de files (pacients). Aquest desequilibri fa que els models tendeixin al *overfitting* i trobin patrons falsos.

Per superar-ho, hem hagut de realitzar un **Anàlisi Exploratòria de Dades (EDA)** extremadament minuciós i quirúrgic amb tres objectius clau per evitar biaixos:
1.  **Evitar el Data Leakage**: Detectar i eliminar variables que, subtilment, revelaven informació del futur (com tractaments que només s'apliquen si hi ha recaiguda), la qual cosa hauria falsejat les mètriques d'èxit.
2.  **Filtratge de Soroll i Feasibility**: Descartar variables que, tot i ser a la base de dades, eren "soroll" estadístic. També hem descartat variables que no eren rellevants per a l'estudi en concret, com metadades referents a la base de dades i als hospitals on s'ha realitzat la cirurgia i recollit les dades.
3.  **Gestió de la Sparsity**: Tractar variables amb un alt percentatge de valors nuls, decidint cas per cas si valia la pena imputar-les (per la seva rellevància clínica, com els receptors hormonals) i amb quina estratègia (per exemple, mitjançant la mitjana de la variable o la moda). En casos molt extrems, que es poden combinar amb altres variables, podem eliminar-les directament.

## Fites de les quals estem orgullosos
Estem especialment orgullosos de la rigorositat del nostre **preprocessat i anàlisi exploratòria (EDA)**. No ens hem limitat a netejar dades superficialment; hem transformat un base de dades amb alt soroll en una font de coneixement sòlida per al modelatge. Hem dedicat un gran esforç a recuperar informació valuosa mitjançant estratègies d'imputació personalitzades per a cada variable, utilitzant la lògica clínica (i no només estadística) per omplir buits crítics sense introduir biaix.

A més, volem destacar la **interpretabilitat** de la nostra solució. Hem aconseguit que el model no funcioni com una "caixa negra", sinó com una eina transparent. Això es veu reflectit en la nostra visualització a **Streamlit**, on els metges no només veuen una probabilitat, sinó que poden entendre quins factors específics (com els estatuts hormonals o característiques del tumor) estan influint en la predicció de risc per al pacient en concret, gràcies a la nostra integració de tècniques explicatives.

## Què hem après
Principalment, hem adquirit un coneixement profund sobre la naturalesa clínica del **càncer d'endometri** i el subgrup molecular NSMP, un àmbit totalment nou per nosaltres, ja que cap membre de l'equip té formació mèdica. Hem après a identificar la rellevància pronòstica de variables com la **invasió miometrial**, els diferents **tipus histològics** o el paper clau dels **receptors hormonals** en la resposta al tractament. Aquesta immersió ha estat essencial per poder distingir entre variables rellevants per al pronòstic i aquelles que no aporten informació valuosa per al model.

A nivell tècnic, hem aprofundit en l'ús de tècniques d'**Explicabilitat (XAI)** com **SHAP**. També hem perfeccionat les nostres habilitats amb **Streamlit**, aprenent a gestionar estats complexos (`session_state`) i a dissenyar interfícies modulars pensades per a l'ús en consultes mèdiques.

## El futur d'Endorisk
El següent pas crític per a la maduresa del projecte és **augmentar el volum de pacients estudiats**. Una base de dades més gran ens permetria:
*   Millorar la robustesa de les prediccions, reduint encara més el risc d'overfitting causat per l'alta dimensionalitat.
*   Extreure conclusions clíniques més sòlides sobre subgrups minoritaris de pacients, donant més fiabilitat estadística als resultats.

A més, si disposéssim de més temps, ens agradaria implementar:
*   **Validació Externa Multicèntrica**: Testejar el model amb dades d'altres hospitals per assegurar la seva generalització i evitar biaixos locals.
*   **Integració amb Històries Clíniques (EHR)**: Automatitzar l'entrada de dades connectant directament l'API de l'app amb el programari hospitalari existent.

---

## Instal·lació i Execució

### Opció 1: Execució Local

```bash
# Clonar repositori
git clone https://github.com/joelmarquez05/bitsXlamarato-2025-pau_overfitting.git
cd bitsXlamarato-2025-pau_overfitting

# Crear entorn virtual i instal·lar dependències
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

# Executar l'aplicació
streamlit run app/main.py
```

### Opció 2: Docker

```bash
# Construir i executar amb Docker Compose
docker-compose up --build

# L'aplicació estarà disponible a: http://localhost:8501
```

Per aturar el contenidor:
```bash
docker-compose down
```
