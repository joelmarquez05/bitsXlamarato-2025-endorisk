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
   - Interpretabilitat (gràfic d'importància de features)
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
