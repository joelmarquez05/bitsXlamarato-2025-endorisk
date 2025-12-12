# CONTEXTO DE DATOS: CÁNCER DE ENDOMETRIO (COHORTE NSMP)

## 1. DESCRIPCIÓN GENERAL DEL DATASET
Este dataset contiene datos clínicos, patológicos, moleculares y de seguimiento de pacientes con cáncer de endometrio.
**Objetivo del Agente:** Predecir el riesgo de recidiva en pacientes pertenecientes exclusivamente al grupo molecular **NSMP** (No Specific Molecular Profile / p53 wild-type).

**IMPORTANTE:** El archivo original (`raw`) contiene pacientes de TODOS los grupos moleculares. El agente debe filtrar primero para obtener la cohorte NSMP limpia.

---

## 2. DICCIONARIO DE VARIABLES Y CODIFICACIÓN

### A. Identificadores y Fechas
*   `codigo_participante`: ID único anonimizado de la paciente.
*   `f_diag`: Fecha de diagnóstico.
*   `fecha_de_recidi`: Fecha en la que se detectó la recidiva.
*   `f_muerte`: Fecha de defunción (si aplica).
*   `Ultima_fecha`: Fecha del último seguimiento o contacto.
*   `diferencia_dias_reci_exit`: Tiempo (en días) entre recidiva y exitus (muerte).

### B. Variables Objetivo (Targets)
*   **`recidiva`** (Target Principal): Variable binaria.
    *   `0`: No Recidiva (Control).
    *   `1`: Recidiva (Caso).
*   `recidiva_exitus`: Estado vital post-recidiva.
*   `libre_enferm`: Tiempo libre de enfermedad (en meses o días, verificar escala).

### C. Clasificación Molecular (CRÍTICO PARA FILTRADO)
Las siguientes variables definen el grupo molecular. Para ser **NSMP**, la paciente debe cumplir **todas** las siguientes condiciones:
1.  **`mut_pole`**: Estado del gen POLE.
    *   `0`: No mutado (Wild Type).
    *   `1`: Mutado (Pathogenic).
    *   **Regla NSMP:** Debe ser `0`.
2.  **`p53_ihq`**: Inmunohistoquímica de p53.
    *   `0`: Wild Type (Normal).
    *   `1`: Overexpression (Anormal).
    *   `2`: Null / Mutated (Anormal).
    *   **Regla NSMP:** Debe ser `0` (Wild Type).
3.  **`mlh1`, `msh2`, `msh6`, `pms2`**: Proteínas de reparación (Mismatch Repair).
    *   `0` / `Intacto`: Presencia de expresión.
    *   `1` / `Perdida`: Pérdida de expresión (MSI).
    *   **Regla NSMP:** Todas deben estar intactas (No MSI-High).

### D. Variables Clínico-Patológicas (Features Predictivas)
*   **`edad`**: Edad al diagnóstico (Numérica).
*   **`imc`**: Índice de Masa Corporal. (Numérica).
*   **`tipo_histologico`**: Tipo de tumor.
    *   `1`: Endometrioide (Más común en NSMP).
    *   `2`: Seroso (Alto riesgo, raro en NSMP puro).
    *   `3`: Células Claras.
    *   `8`: Carcinosarcoma.
    *   *Nota:* Si el agente detecta `8` (Carcinosarcoma) o Sarcomas en texto libre, excluir del análisis NSMP.
*   **`Grado`**: Grado de diferenciación tumoral.
    *   `1`: Bien diferenciado.
    *   `2`: Moderadamente diferenciado.
    *   `3`: Pobremente diferenciado (Alto grado).
*   **`infiltracion_mi`**: Invasión del miometrio.
    *   `1`: < 50%.
    *   `2`: ≥ 50%.
*   **`infilt_estr_cervix`**: Invasión estroma cervical (0=No, 1=Sí).
*   **`inf_param_vag`**: Invasión parametrios/vagina (0=No, 1=Sí).
*   **`afectacion_linf`**: Ganglios positivos (0=No, 1=Sí).
*   **`valor_de_ca125`**: Marcador tumoral sérico pre-operatorio (Numérica).
*   **`estadiaje_pre_i`**: Estadificación por imagen preoperatoria.
*   **`FIGO2023`**: Estadio FIGO definitivo (Categoría ordinal: I, II, III, IV).

### E. Tratamiento (Variables de Confusión)
*   **`tto_1_quirugico`**: Tipo de cirugía (Histerectomía, linfadenectomía, etc.).
*   **`Tratamiento_RT`**: Radioterapia.
    *   `0`: No.
    *   `1`: Sí (Externa).
    *   `2`: Braquiterapia.
*   **`Tratamiento_sistemico`**: Quimioterapia.
    *   `0`: No.
    *   `1`: Sí.

---

## 3. REGLAS DE LIMPIEZA Y PREPROCESAMIENTO PARA EL AGENTE

### 1. Filtro de Cohorte (NSMP)
El agente **DEBE** ejecutar este filtro antes de cualquier entrenamiento:
```python
df_nsmp = df[
    (df['mut_pole'] == 0) & 
    (df['p53_ihq'] == 0) & 
    (df['tipo_histologico'] != 8) & # Excluir Carcinosarcomas
    (df['tipo_histologico'] != 9)   # Excluir Sarcomas/Otros no epiteliales
]
```

### 2. Manejo de Valores Nulos (NA)
*   **`valor_de_ca125`**: Si es `NA`, imputar con mediana (o marcar como desconocido si el % es alto).
*   **`grado`**: Si es `NA` y el tipo es "Endometrioide", verificar columna `comentarios`.
*   **`imc`**: Valores faltantes pueden imputarse con la media de la cohorte.

### 3. Exclusiones por Texto Libre
El agente debe escanear la columna **`comentarios`** o **`otra_histo`** buscando las siguientes palabras clave para **EXCLUIR** pacientes mal clasificadas:
*   "Sarcoma"
*   "Leiomiosarcoma"
*   "POLE mut" (si la columna numérica estaba vacía pero el texto lo confirma)
*   "p53 mutado" / "p53 abn"

---

## 4. RELACIÓN DE ARCHIVOS
*   **CSV Fuente:** `IQ_Cancer_Endometrio_merged_NMS.csv`
*   **Separador:** Coma (`,`) o Punto y coma (`;`) (El agente debe detectar esto automáticamente).
*   **Codificación:** UTF-8.

---

## 5. NOTAS CLÍNICAS DE IMPORTANCIA
*   El grupo **NSMP** suele tener un pronóstico intermedio. La predicción de recidiva en este grupo es más difícil que en los extremos (POLE o p53abn), por lo que variables sutiles como `infiltracion_mi`, `Grado` y `invasión linfo-vascular` (LVSI) cobran mayor relevancia.
*   Las pacientes con **Estadio IV** (Metástasis a distancia `metasta_distan`=1) tienen un riesgo de recidiva/muerte cercano al 100%, considerar si se incluyen en el modelo o si sesgan el entrenamiento para estadios precoces.