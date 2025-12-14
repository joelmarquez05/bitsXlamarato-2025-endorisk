# Instruccions per a l'Agent d'IA - Projecte NEST

## Resum del Projecte
**Nom:** NEST (NSMP Endometrial Stratification Tool)
**Objectiu:** Predir la **probabilitat de recaiguda** en pacients amb Càncer d'Endometri NSMP.
**Usuaris finals:** Oncòlegs Ginecològics.

## Producte Final
Aplicació Streamlit amb:
1. **Login** - Autenticació del doctor
2. **Formulari** - Entrada de dades del pacient
3. **Resultat** amb 3 panells:
   - Predicció (probabilitat + semàfor 🟢🟡🔴)
   - Interpretabilitat (gràfic d'importància de features, tècniques com SHAP)
   - Explicació IA (Gemini API)
4. **Disseny Responsive** - L'app ha de visualitzar-se correctament en dispositius mòbils (ytelèfons)

## Coneixement Mèdic Crític
- **NSMP** = 50% dels casos, grup "calaix de sastre"
- **Prioritzem Sensibilitat (Recall)** - Un Fals Negatiu és fatal
- Divisió en: Baix, Mitjà i Alt Risc

## Stack Tècnic
- Python 3.11 / Streamlit / Scikit-learn / Pandas / Joblib
- Google Gemini API (explicacions)
- Docker & Docker Compose

## Estructura de Fitxers
```
├── data/raw/           # Dades originals
├── data/processed/     # Dades netes
├── notebooks/          # 01→02→03→04 en ordre
├── models/             # Models entrenats (.joblib)
└── app/                # Streamlit app
```

---

## Regles Obligatòries per a l'IA

### 1. Notebooks (.ipynb)
**MAI editis un fitxer `.ipynb` directament.**
Només proporciona el codi perquè l'usuari el copiï.

### 2. Comentaris
- **Minimitza els comentaris** - Només quan sigui estrictament necessari
- **Tots els comentaris en CATALÀ**
- Exemple:
```python
# Carreguem el model entrenat
model = joblib.load('models/model.joblib')

# Fem la predicció
prob = model.predict_proba(X)[0][1]
```

### 3. Idioma del Codi
| Element | Idioma |
|---------|--------|
| Variables i funcions | Anglès |
| Comentaris | Català |
| UI Strings | Castellà/Català |

### 4. Convencions de Naming

#### Variables
```python
# ✅ Correcte: snake_case, descriptiu, anglès
patient_age = 65
tumor_grade = 3
myometrial_invasion = 0.6
has_lvsi = True
recurrence_probability = 0.78

# ❌ Incorrecte
edad = 65           # Castellà
tg = 3              # Massa curt
myometrialInvasion  # camelCase
```

#### Funcions
```python
# ✅ Correcte: snake_case, verb + noun
def load_patient_data(filepath):
    pass

def calculate_recurrence_risk(features):
    pass

def preprocess_clinical_features(df):
    pass

# ❌ Incorrecte
def calcularRiesgo():    # Castellà + camelCase
def data():              # Massa genèric
def process():           # Què processa?
```

#### Classes
```python
# ✅ Correcte: PascalCase
class PatientData:
    pass

class RecurrencePredictor:
    pass

class FeatureEncoder:
    pass
```

#### Constants
```python
# ✅ Correcte: SCREAMING_SNAKE_CASE
TARGET_COLUMN = 'recurrence'
RISK_THRESHOLDS = {'low': 0.3, 'medium': 0.6, 'high': 0.8}
MODEL_PATH = 'models/model.joblib'
GEMINI_MODEL = 'gemini-pro'
```

#### Fitxers
| Tipus | Format | Exemple |
|-------|--------|---------|
| Notebooks | `##_descriptive_name.ipynb` | `03_model_training.ipynb` |
| Mòduls Python | `snake_case.py` | `feature_engineering.py` |
| Models | `model_name.joblib` | `random_forest_v2.joblib` |
| Dades | `descriptive_name.csv` | `features_train.csv` |

#### Prefixos i Sufixos Estàndard
| Prefix/Sufix | Significat | Exemple |
|--------------|------------|---------|
| `X_`, `y_` | Features / Target | `X_train`, `y_test` |
| `_df` | DataFrame | `patients_df` |
| `_path` | Ruta de fitxer | `model_path` |
| `_config` | Configuració | `app_config` |
| `is_`, `has_` | Booleans | `is_valid`, `has_lvsi` |
| `n_`, `num_` | Comptadors | `n_samples`, `num_features` |
| `_scaled` | Dades normalitzades | `X_scaled` |
| `_encoded` | Dades codificades | `stage_encoded` |

#### Convencions de Versionat

Quan es fan múltiples experiments (preprocessats, models, etc.), segueix aquest format:

**Format:** `nom_descriptiu_vX.extensió`

| Tipus | Exemple | Descripció |
|-------|---------|------------|
| **Models** | `random_forest_v1.joblib` | Versió 1 del RF |
|  | `xgboost_smote_v2.joblib` | XGBoost amb SMOTE, versió 2 |
| **Scalers** | `scaler_v1.joblib` | Scaler per a v1 del model |
|  | `scaler_minmax_v2.joblib` | MinMaxScaler per a v2 |
| **Dades processades** | `features_train_v1.csv` | Train set versió 1 |
|  | `features_train_smote_v2.csv` | Amb SMOTE aplicat |
| **Encoders** | `label_encoder_v1.joblib` | Encoder per a categoricals |

**Regles:**
- Sempre incrementa la versió quan canviïs el preprocessat o els hiperparàmetres
- Inclou tècnica en el nom si és rellevant (`_smote`, `_minmax`, `_pca`)
- El model i el scaler han de coincidir en versió (`model_v2` + `scaler_v2`)
- MAI sobreescriguis versions anteriors, crea una nova

**Estructura de directoris recomanada:**
```
models/
├── model_v1.joblib           # Baseline
├── model_v2.joblib           # Amb tuning
├── model_smote_v3.joblib     # Amb SMOTE
├── scaler_v1.joblib
├── scaler_v2.joblib
└── scaler_v3.joblib

data/processed/
├── features_train_v1.csv
├── features_test_v1.csv
├── features_train_v2.csv     # Nou preprocessat
└── features_test_v2.csv
```

### 5. Estil de Codi
- Segueix PEP 8
- Codi net i llegible
- Màxim 88 caràcters per línia
- Evita codi redundant

### 6. Convencions de Fitxers

#### Imports (ordre)
```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
```

#### Estructura d'un mòdul
```python
"""
Descripció breu del mòdul (en anglès).
"""
# Imports
import pandas as pd

# Constants
TARGET_COLUMN = 'recurrence'

# Funcions auxiliars (privades amb _)
def _validate_input(df):
    pass

# Funcions públiques
def load_data(path):
    pass

# Classes
class DataProcessor:
    pass
```

### 7. Vocabulari del Domini (Terminologia Mèdica)

Utilitza sempre aquests termes en anglès per consistència:

| Català/Castellà | Anglès (usar sempre) |
|-----------------|---------------------|
| Recaiguda/Recaída | `recurrence` |
| Grau tumoral | `tumor_grade` |
| Invasió miometrial | `myometrial_invasion` |
| Estadi | `stage` |
| LVSI | `lvsi` (ja és acrònim anglès) |
| Pacient | `patient` |
| Predicció | `prediction` |
| Probabilitat | `probability` |
| Risc | `risk` |
| Alt/Mitjà/Baix | `high`, `medium`, `low` |

---

## Informació Tècnica del Projecte

### Workflow de Notebooks (Executar en ORDRE)

```
01_exploratory_analysis.ipynb  → No guarda res
02_data_preprocessing.ipynb    → GUARDA: data/processed/*.csv, models/scaler.joblib
03_model_training.ipynb        → CARGA des de data/processed/, GUARDA: models/model.joblib
04_model_evaluation.ipynb      → CARGA: models/model.joblib, features_test.csv
app/main.py (Streamlit)        → CARGA: models/model.joblib, models/scaler.joblib
```

### Joblib (Persistència de Models)

Guardar model/scaler:
```python
import joblib
joblib.dump(model, '../models/model.joblib')
joblib.dump(scaler, '../models/scaler.joblib')
```

Carregar en app o notebook posterior:
```python
model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')
```

### Entorns d'Execució

| Entorn | Ús | Comanda |
|--------|-----|---------|
| **Docker** | App Streamlit | `docker compose up --build -d` |
| **venv local** | Notebooks | `.venv\Scripts\activate` |

### Paths Importants (des de notebooks/)
- Dades: `../data/processed/`
- Models: `../models/`
- Scaler: `../models/scaler.joblib`

### Paths Importants (des de app/)
- Models: `models/` (sense `../` perquè Docker munta a /app)

---

# Context del Projecte: Predicció de Recaiguda en Càncer d'Endometri (Perfil NSMP)

**Objectiu:** Entrenar un model de Machine Learning (Classificació Binària) per predir la variable `recidiva`.
**Dataset:** `data/raw/cancer_endometri.csv` - Dades clíniques, patològiques i moleculars de pacients amb càncer d'endometri.
**Focus:** Especial interès en el perfil molecular **NSMP** (Non-Specific Molecular Profile), on els receptors hormonals són clau.

---

## 1. Variable Objectiu (Target) 🎯

*   **`recidiva`**:
    *   **0**: No (Pacient lliure de malaltia).
    *   **1**: Sí (La malaltia va reaparèixer).
    *   **Acció:** Aquesta és l'etiqueta a predir (`y`).
    *   *Nota:* Eliminar files on el valor sigui `2` (Desconegut) o `NA`.

---

## 2. Variables Predictores (Features) ✅

Aquestes són les columnes que contenen informació vàlida (clínica, patològica o molecular) disponible després de la cirurgia inicial i que has d'usar per entrenar el model (`X`).

### A. Perfil Pacient
*   **`edad`**: Numèrica. Edat al diagnòstic.
*   **`imc`**: Numèrica. Índex de Massa Corporal (Obesitat és factor de risc).
*   **`asa`**: Categòrica/Ordinal (1-4). Estat físic de la pacient (comorbiditats).

### B. Característiques del Tumor (Patologia)
*   **`histo_defin`** (o `tipo_histologico`): Categòrica. Tipus de càncer (Endometrioide, Serós, etc.). *Crucial*.
*   **`grado_histologi`** (o `Grado`): Categòrica/Ordinal.
    *   `1`: Baix grau (G1-G2).
    *   `2`: Alt grau (G3). *Factor de risc alt*.
*   **`tamano_tumoral`**: Numèrica. Mida en cm.
*   **`infiltracion_mi`**: Categòrica/Ordinal. Profunditat d'invasió en l'úter.
    *   `0`: No, `1`: <50%, `2`: >50%, `3`: Serosa. *A major valor, major risc*.
*   **`afectacion_linf`** (LVSI): Binària (0/1). Invasió Limfovascular. *Factor pronòstic crític*.
*   **`infilt_estr_cervix`**: Binària (0/1). Si el tumor ha baixat al coll uterí.
*   **`FIGO2023`**: Categòrica (Estadiatge). Resumeix la gravetat.
    *   **Transformació:** Convertir a escala ordinal (IA1=1, IA2=2 ... IVC=14).

### C. Perfil Molecular (Clau per a NSMP) 🧬
*   **`recep_est_porcent`**: Numèrica (0-100). **Variable Estrella**. Percentatge de Receptors d'Estrogen.
    *   *Hipòtesi:* En NSMP, alt percentatge = millor pronòstic.
*   **`rece_de_Ppor`**: Numèrica (0-100). Receptors de Progesterona.
*   **`p53_ihq`**: Categòrica. (1=Normal, 2=Anormal/Mutat). *Defineix subtipus*.
*   **`mut_pole`**: Categòrica. (1=Mutat, 2=No mutat). *Defineix subtipus*.
*   **`msh2`, `msh6`, `pms2`, `mlh1`**: Categòriques. Proteïnes de reparació (Mismatch Repair). Si alguna és anormal (1), és perfil MMRd.

### D. Dades Quirúrgiques i Tractament Inicial
*   **`n_GC_Afect`**: Numèrica. Nombre de ganglis sentinella afectats.
*   **`Reseccion_macroscopica_complet`**: Binària (0/1). Si s'ha extret tot el tumor visible (`1` és l'ideal).
*   **`Tratamiento_RT`** / **`Tratamiento_sistemico`**: Binàries. Si va rebre Radioteràpia o Quimio adjuvant.

---

## 3. Variables a ELIMINAR (Data Leakage / Soroll) 🛑

Aquestes variables contenen informació del "futur" (ocorren quan ja sabem que ha recaigut) o són identificadors irrellevants. **No les usis en l'entrenament.**

### Data Leakage (Spoilers del futur)
*   `recidiva_exitus`: És un proxy del target.
*   `diferencia_dias_reci_exit`: Revela quant temps va viure/va estar sana.
*   `fecha_de_recidi`: Si té data, és que va recaure.
*   `f_muerte`: Revela si va morir.
*   `causa_muerte`: Revela el resultat.
*   `tto_recidiva`: Tractament rebut *per* la recaiguda.
*   `Tt_recidiva_qx`: Cirurgia *per* la recaiguda.
*   `loc_recidiva_r01` a `r06`: Lloc de la recaiguda.
*   `numero_de_recid`, `num_recidiva`: Quantitat de recaigudes.
*   `libre_enferm`: Estat actual (similar al target).
*   `est_pcte`: Estat actual (viva/morta).
*   `visita_control`, `Ultima_fecha`: Dates de seguiment.

### Soroll / Identificadors
*   `codigo_participante`: ID aleatori.
*   `usuario_reg1`: ID administratiu.
*   Dates crues (`f_diag`, `FN`, `fecha_qx`): Usar `edad` en el seu lloc.
*   `comentarios`, `otras_especifi`: Text lliure no estructurat (difícil de processar amb poques dades).
*   `compl_precoc_rXX` (Complicacions): Generalment no prediuen la biologia del càncer, només el postoperatori immediat.

---

## 4. Instruccions de Preprocessat ⚙️

1.  **Filtratge de files:** Elimina files on `recidiva` sigui `NA` o `2` (Desconegut).
2.  **Imputació de Nuls (`NA`):**
    *   Per `recep_est_porcent`: Si falta, és crític. Intentar imputar amb la mitjana o marcar com a "Desconegut".
    *   Per `Grado` o `FIGO`: Imputar amb la moda (valor més freqüent).
3.  **Codificació:**
    *   Variables binàries (Sí/No) -> 1/0.
    *   `FIGO2023`: Mapejar manualment a ordre numèric (IA1 < IA2 < IB ...).
4.  **Estratègia NSMP:**
    *   Idealment, el model hauria d'aprendre que si `p53_ihq` és Normal i `mut_pole` és No Mutat, la variable `recep_est_porcent` guanya molt de pes predictiu.

