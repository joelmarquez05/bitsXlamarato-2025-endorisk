# Guía del Proyecto NEST - Para el Equipo

---

## 🚀 Quick Start (Lo Primero)

### App Streamlit (Docker)
```bash
docker compose up --build -d
```
→ Abre **http://localhost:8501**

### Notebooks (venv local)
```bash
# Primera vez:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Cada vez que abras el proyecto:
.venv\Scripts\activate
```

### API Key (.env)
Crea archivo `.env` en la raíz:
```
GEMINI_API_KEY=tu_api_key_aqui
```

---

## 1. ¿Qué estamos construyendo?

**NEST** = Herramienta para predecir la probabilidad de recaída en pacientes con cáncer de endometrio NSMP.

### Producto Final
Una app web (Streamlit) donde:
1. El doctor hace login
2. Introduce datos del paciente
3. Ve la predicción + explicación de por qué el modelo predijo eso

---

## 2. ¿Por qué dividimos el trabajo en notebooks?

### El problema de "todo en una notebook"
Cuando trabajamos todo en una sola notebook:
- ❌ El archivo se hace enorme e inmanejable
- ❌ Si alguien cambia algo, puede romper todo
- ❌ No podemos reutilizar el modelo entrenado en la app
- ❌ Cada vez hay que re-entrenar todo desde cero

### La solución: Dividir y Guardar
```
Notebook 01: Explorar datos → No guarda nada
Notebook 02: Limpiar datos  → GUARDA archivos CSV
Notebook 03: Entrenar modelo → GUARDA modelo .joblib
Notebook 04: Evaluar modelo → CARGA el modelo guardado
App Streamlit              → CARGA el modelo y hace predicciones
```

**Ventaja:** Entrenas el modelo UNA VEZ, lo guardas, y la app lo usa sin re-entrenar.

---

## 3. ¿Qué es Joblib y por qué lo usamos?

**Joblib** es una librería que permite guardar objetos de Python (como modelos de ML) en archivos.

### ¿Por qué no podemos simplemente "pasarle" el modelo a la app?
Porque las notebooks y la app son procesos separados. No comparten memoria.

### Analogía
Imagina que el modelo es un documento de Word:
- Sin guardar: Solo existe mientras Word está abierto
- Guardado (.docx): Puedes cerrarlo y abrirlo luego, o enviarlo a otro PC

Con modelos de ML es igual:
- Sin guardar: Solo existe mientras la notebook está ejecutándose
- Guardado (.joblib): Puedes cerrar la notebook, y la app puede cargarlo

---

## 4. Tutorial: Cómo Guardar y Cargar

### Instalación
```python
# Ya incluido en requirements.txt, pero por si acaso:
pip install joblib
```

### Guardar un modelo (en notebook 03)
```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Entrenas tu modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Lo guardas en un archivo
joblib.dump(model, '../models/model.joblib')
print("Modelo guardado!")
```

### Cargar un modelo (en la app o notebook 04)
```python
import joblib

# Cargas el modelo desde el archivo
model = joblib.load('../models/model.joblib')

# Ya puedes usarlo para predecir
prediction = model.predict(X_new)
probabilidad = model.predict_proba(X_new)[0][1]
```

---

## 5. ¿Qué más se puede guardar con Joblib?

| Objeto | ¿Guardarlo? | Archivo sugerido |
|--------|-------------|------------------|
| Modelo entrenado | ✅ Sí | `models/model.joblib` |
| Scaler (StandardScaler, etc.) | ✅ Sí | `models/scaler.joblib` |
| Encoder (LabelEncoder, etc.) | ✅ Sí | `models/encoder.joblib` |
| DataFrame procesado | ❌ No (usar CSV) | `data/processed/datos.csv` |

### ¿Por qué guardar el Scaler?
Si normalizas los datos antes de entrenar, necesitas la MISMA normalización para predecir.

```python
# En notebook 02 (preprocesamiento)
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Ajusta y transforma
joblib.dump(scaler, '../models/scaler.joblib')  # Guarda

# En la app (predicción)
import joblib

scaler = joblib.load('models/scaler.joblib')  # Carga
X_new_scaled = scaler.transform(X_new)  # Solo transforma (NO fit!)
prediction = model.predict(X_new_scaled)
```

---

## 6. Estructura del Proyecto

```
bitsXlamarato-2025-pau_overfitting/
│
├── 📁 data/
│   ├── raw/                    # Datos originales
│   └── processed/              # Datos limpios (CSV)
│
├── 📁 notebooks/               # Ejecutar EN ORDEN: 01 → 02 → 03 → 04
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb   # Guarda → data/processed/
│   ├── 03_model_training.ipynb       # Guarda → models/
│   └── 04_model_evaluation.ipynb
│
├── 📁 models/                  # Aquí se guardan los .joblib
│   ├── model.joblib
│   └── scaler.joblib
│
├── 📁 app/                     # App Streamlit
│   └── main.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── agent.MD                    # Instrucciones para la IA
└── guia.MD                     # Este archivo (para el equipo)
```

---

## 7. Flujo de Trabajo Paso a Paso

### Día 1: Preparación
```
1. Clonar el repo
2. docker-compose up --build
3. Abrir notebooks en JupyterLab o VS Code
```

### Flujo de Notebooks
```
┌─────────────────────────────────────────────────────────────┐
│  01_exploratory_analysis.ipynb                              │
│  - Ver distribuciones, valores nulos, correlaciones         │
│  - NO guarda nada                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  02_data_preprocessing.ipynb                                │
│  - Limpiar datos, crear features, split train/test          │
│  - GUARDA:                                                  │
│    • data/processed/features_train.csv                      │
│    • data/processed/features_test.csv                       │
│    • models/scaler.joblib                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  03_model_training.ipynb                                    │
│  - CARGA: data/processed/features_train.csv                 │
│  - Entrenar modelo(s), comparar métricas                    │
│  - GUARDA: models/model.joblib                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  04_model_evaluation.ipynb                                  │
│  - CARGA: models/model.joblib, features_test.csv            │
│  - Evaluar en test, matrices de confusión, métricas         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  app/main.py (Streamlit)                                    │
│  - CARGA: models/model.joblib, models/scaler.joblib         │
│  - Recibe input del doctor                                  │
│  - Muestra predicción + explicación (Gemini API)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Setup del Entorno

Tenemos **DOS entornos** separados:

| Entorno | Uso | ¿Cuándo? |
|---------|-----|----------|
| **Docker** | App Streamlit | Siempre para ver/probar la app |
| **venv local** | Notebooks Jupyter | Para desarrollo de ML |

---

### 8.1 Docker (App Streamlit)

**Único comando que necesitas:** 
```bash
docker compose up --build -d
```

> 💡 El flag `-d` hace que corra en segundo plano (no bloquea tu terminal)

La app estará disponible en: **http://localhost:8501**

#### Otros comandos útiles:
```bash
# Ver logs del contenedor
docker compose logs -f

# Parar el contenedor
docker compose down

# Reconstruir si cambiaste algo
docker compose up --build -d
```

---

### 8.2 Entorno Virtual Local (Notebooks)

Para trabajar con las notebooks necesitas un entorno Python local.

#### Primera vez (crear el venv):
```bash
# Crear el entorno virtual
python -m venv .venv

# Activar el entorno (Windows PowerShell)
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

#### Cada vez que abras el proyecto:
```bash
# Solo necesitas activar el venv
.venv\Scripts\activate
```

> 📍 Sabrás que está activo cuando veas `(.venv)` al inicio de tu terminal

#### En VS Code:
1. Abre una notebook
2. Click en "Select Kernel" (arriba a la derecha)
3. Selecciona `.venv` o "Python 3.x (.venv)"

---

### 8.3 Archivo `.env` (API Keys)

Crea un archivo llamado `.env` en la raíz del proyecto:

```
GEMINI_API_KEY=tu_api_key_real_aqui
```

> 📍 Obtener API key: https://makersuite.google.com/app/apikey

⚠️ **IMPORTANTE:** Este archivo está en `.gitignore` y NO se sube a GitHub.

---

## 9. Resumen Rápido

| Acción | Comando/Código |
|--------|----------------|
| Guardar modelo | `joblib.dump(model, 'models/model.joblib')` |
| Cargar modelo | `model = joblib.load('models/model.joblib')` |
| Guardar CSV | `df.to_csv('data/processed/datos.csv', index=False)` |
| Cargar CSV | `df = pd.read_csv('data/processed/datos.csv')` |
| Levantar Docker | `docker-compose up --build` |
| Ejecutar app | `streamlit run app/main.py` |

---

## 10. Preguntas Frecuentes

### ¿Por qué los paths tienen `../`?
Porque las notebooks están en `notebooks/`, y necesitan "subir" un nivel para acceder a `models/` o `data/`.

```
notebooks/03_training.ipynb  →  quiere guardar en  →  models/model.joblib
                ↓ sube un nivel
       ../models/model.joblib  ✓
```

### ¿Qué pasa si modifico el preprocesamiento?
Tienes que re-ejecutar notebooks 02, 03, 04 en orden. El modelo antiguo ya no será válido.

### ¿El modelo se guarda automáticamente?
NO. Tienes que ejecutar la celda con `joblib.dump()` explícitamente.
