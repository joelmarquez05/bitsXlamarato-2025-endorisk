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
├── app/                # Streamlit app
└── src/                # Mòduls reutilitzables
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

# 3. Local
from src.preprocessing import clean_data
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
