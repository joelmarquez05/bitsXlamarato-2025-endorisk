Aquí tienes la transcripción completa de las variables y sus diccionarios de valores, basada estrictamente en los documentos PDF que has proporcionado.

He agrupado las variables por las secciones del documento original para facilitar su lectura.

# Diccionario de Datos: Cáncer de Endometrio

## 1. Diagnóstico e Histología Inicial

| Variable | Valor | Significado |
| :--- | :--- | :--- |
| **`despues_diag`** | 0 | No (No decide tratarse en el hospital) |
| | 1 | Sí (Decide tratarse en el hospital) |
| **`ecotv_infiltobj`** | 1 | No aplicado |
| *(Método Objetivo)* | 2 | < 50% |
| | 3 | > 50% |
| | 4 | No valorable |
| **`ecotv_infiltsub`** | 1 | No aplicado |
| *(Método Subjetivo)* | 2 | < 50% |
| | 3 | > 50% |
| | 4 | No valorable |
| **`estadiaje_pre_i`** | 0 | Estadio I |
| *(Pre-Quirúrgico)* | 1 | Estadio II |
| | 2 | Estadio III y IV |
| **`Grado`** | 1 | Bajo grado (G1-G2) |
| | 2 | Alto grado (G3) |
| **`grupo_riesgo`** | 1 | Riesgo bajo |
| *(Preoperatorio)* | 2 | Riesgo intermedio |
| | 3 | Riesgo alto |
| **`metasta_distan`** | 0 | No |
| | 1 | Sí |
| **`tipo_histologico`** | 1 | Hiperplasia con atipias |
| | 2 | Carcinoma endometrioide |
| | 3 | Carcinoma seroso |
| | 4 | Carcinoma Células claras |
| | 5 | Carcinoma Indiferenciado |
| | 6 | Carcinoma Mixto |
| | 7 | Carcinoma Escamoso |
| | 8 | Carcinosarcoma |
| | 9 | Leiomiosarcoma |
| | 10 | Sarcoma de estroma endometrial |
| | 11 | Sarcoma indiferenciado |
| | 12 | Adenosarcoma |
| | 88 | Otros |

---

## 2. Tratamiento Quirúrgico (Técnica)

| Variable | Valor | Significado |
| :--- | :--- | :--- |
| **`abordajeqx`** | 1 | Laparoscopia |
| | 2 | Laparotomía |
| | 3 | Robótica |
| | 4 | Vaginal |
| **`afectacion_linf`** | 0 | No |
| | 1 | Sí |
| **`afectacion_omen`** | 0 | No |
| | 1 | Sí |
| **`Anexectomia`** | 0 | No |
| | 1 | Sí |
| **`asa`** | 0 - 5 | Grados ASA 1 a ASA 6 |
| | 6 | Desconocido |
| **`conver_laparo`** | 0 | No |
| | 1 | Sí |
| **`hsp_trat_primario`** | 0 | No |
| | 1 | Sí |
| **`presntado_cTG`** | 0 | No |
| *(Comité Tumores)* | 1 | Sí |
| | 2 | Se desconoce |
| **`tto_1_quirugico`** | 0 | No |
| | 1 | Sí |
| **`tto_NA`** | 0 | No (Tratamiento Neoadyuvante) |
| | 1 | Sí |
| **`perforacion_uterina`** | 0 | No |
| | 1 | Sí |
| **`movilizador_uterino`** | 0 | No |
| | 1 | Sí |
| | 2 | Se desconoce |
| **`trazador_utiliz`** | 0 | Verde de indocianina |
| | 1 | Otros |
| **`tec_Qx`** | 0 | Inicial |
| | 1 | Avanzado |
| | 2 | Estadiaje |

### Técnicas Quirúrgicas Específicas (Invertidas en el PDF)
*Nota: En estas variables específicas, el PDF indica 0=Sí / 1=No.*

| Variable | Valor | Significado |
| :--- | :--- | :--- |
| **`histe_avanz`** | 0 | Sí |
| | 1 | No |
| | 2 | No aplica |
| **`oment_Avan`** | 0 | Sí |
| | 1 | No |
| **`tec_apendic`** | 0 | Sí |
| | 1 | No |
| | 2 | No aplica |
| **`tec_biop_periAvn`** | 0 | Sí |
| | 1 | No |
| **`tec_especnectom`** | 0 | Sí |
| | 1 | No |
| **`Tec_histerec`** | 0 | No (*Excepción: aquí 0 es No*) |
| | 1 | Sí |
| **`tec_pelviperito`** | 0 | Sí |
| | 1 | No |
| **`tec_rectosigmo`** | 0 | Sí |
| | 1 | No |
| **`tec_strip_diafg`** | 0 | Sí |
| | 1 | No |

---

## 3. Anatomía Patológica (Ganglios y Extensión)

| Variable | Valor | Significado |
| :--- | :--- | :--- |
| **`AP_centinela_pelvico`** | 0 | Negativo (pN0) |
| | 1 | Cels. tumorales aisladas (pN0(i+)) |
| | 2 | Micrometástasis (pN1(mi)) |
| | 3 | Macrometástasis (pN1) |
| | 4 | pNx |
| **`AP_ganPelv`** | 0 | Negativo |
| | 1 | Cels. tumorales aisladas |
| | 2 | Micrometástasis |
| | 3 | Macrometástasis |
| **`AP_glanPaor`** | 0 | Negativo |
| *(Paraórticos)* | 1 | Cels. tumorales aisladas |
| | 2 | Micrometástasis |
| | 3 | Macrometástasis |
| **`ap_gPelv_loc`** | 0 | Izquierda |
| | 1 | Derecha |
| | 2 | Bilateral |
| **`gc_ap`** | 0 | No realizado |
| *(Ganglio Centinela intraop)* | 1 | Negativo |
| | 2 | Positivo |
| **`gc_lpd` / `gc_lpi`** | 0 | No migra |
| *(Lat. Pélvica Der/Izq)* | 1 | Ilíaca |
| | 2 | Obturatriz |
| **`gc_paraor`** | 1 | Interilíacos |
| | 2 | Inframesentéricos |
| | 3 | Supramesentéricos |
| | 9 | No encontrado |
| **`Local_Gan_Paor`** | 1 | Inframesentéricos |
| | 2 | Supramesentéricos |
| | 3 | Infra + Supra |
| **`n_resec_Intes`** | Numérico | Nº de resecciones intestinales |

---

## 4. Anatomía Patológica Definitiva e Inmunohistoquímica

| Variable | Valor | Significado |
| :--- | :--- | :--- |
| **`grado_histologi`** | 1 | Bajo grado (G1-G2) |
| | 2 | Alto grado (G3) |
| **`grupo_de_riesgo_definitivo`**| 1 | Riesgo bajo |
| | 2 | Riesgo intermedio |
| | 3 | Riesgo intermedio-alto |
| | 4 | Riesgo alto |
| | 5 | Avanzados |
| **`histo_defin`** | 1-12 | *Mismos valores que `tipo_histologico`* |
| **`infiltracion_mi`** | 0 | No infiltración |
| | 1 | < 50% |
| | 2 | > 50% |
| | 3 | Infiltración serosa |
| **`infilt_estr_cervix`** | 0 | No |
| | 1 | Sí |
| | 2 | Se desconoce |
| **`inf_param_vag`** | 0 | No |
| | 1 | Parametrio derecho |
| | 2 | Parametrio izquierdo |
| | 3 | Vagina |
| **`tx_anexial`** | 0 | No |
| | 1 | Derecha |
| | 2 | Izquierda |
| | 3 | Bilateral |
| **`tx_sincronico`** | 0 | No |
| | 1 | Sí |

### Marcadores Moleculares
| Variable | Valor | Significado |
| :--- | :--- | :--- |
| **`beta_cateninap`** | 0 | No |
| | 1 | Sí |
| | 2 | No realizado |
| **`mlh1` / `msh2` / `msh6`** | 0 | Normal |
| | 1 | Anormal |
| | 2 | No realizado |
| **`mut_pole`** | 1 | Mutado |
| | 2 | No Mutado |
| | 3 | No realizado |
| **`p53_ihq`** | 1 | Normal |
| | 2 | Anormal |
| | 3 | No realizada |
| **`p53_molecular`** | 1 | Mutado / anormal |
| | 2 | No mutado / wild-type |
| | 3 | No realizada |
| **`pms2`** | 0 | Normal |
| | 1 | Anormal |
| | 2 | No realizado |

### Estudio Genético
*Nota: En el archivo CSV estas opciones aparecen desglosadas en columnas binarias (r01, r02...)*
| Variable | Valor | Significado |
| :--- | :--- | :--- |
| **`estudio_genetico`** | 1 | Negativo |
| | 2 | BRCA1 |
| | 3 | BRCA2 |
| | 4 | Lynch |
| | 5 | Otros |
| | 6 | No realizado |

---

## 5. Estadiaje (FIGO)

| Variable | Valor | Significado |
| :--- | :--- | :--- |
| **`estadificacion_`** | 1 | Ia |
| *(FIGO 2018)* | 2 | Ib |
| | 3 | II |
| | 4 | IIIa |
| | 5 | IIIb |
| | 6 | IIIc1 |
| | 7 | IIIc2 |
| | 8 | IVa |
| | 9 | IVb |
| **`FIGO2023`** | 1 | IA1 |
| | 2 | IA2 |
| | 3 | IA3 |
| | 4 | IB |
| | 5 | IC |
| | 6 | IIA |
| | 7 | IIB |
| | 8 | IIC |
| | 9 | IIIA |
| | 10 | IIIB |
| | 11 | IIIC |
| | 12 | IVA |
| | 13 | IVB |
| | 14 | IVC |

---

## 6. Tratamiento Adyuvante

| Variable | Valor | Significado |
| :--- | :--- | :--- |
| **`inten_tto`** | 1 | Curativo |
| | 2 | Paliativo |
| **`qt`** | 0 | No |
| *(Quimioterapia)* | 1 | Sí |
| **`rdt`** | 0 | RT indicada pero no realizada |
| *(Lugar Radio)* | 1 | Pélvica |
| | 2 | Campo ampliado (pélvica+ParaAo) |
| **`rt_dosis`** | 0 | No realizada |
| | 1 | Dosis parcial |
| | 2 | Dosis completa |
| **`Tratamiento_sistemico_realizad`** | 0 | No realizada |
| | 1 | Dosis parcial |
| | 2 | Dosis completa |
| **`Tributaria_a_Radioterapia`** | 0 | No |
| | 1 | Sí |

---

## 7. Seguimiento y Recidiva (Variables Target / Leakage)

| Variable | Valor | Significado |
| :--- | :--- | :--- |
| **`causa_muerte`** | 0 | Por el cáncer de endometrio |
| | 1 | Otras causas |
| **`dx_recidiva`** | 0 | Clínico |
| | 1 | Prueba complementaria |
| **`est_pcte`** | 1 | Viva |
| | 2 | Muerta |
| | 3 | Desconocido |
| **`libre_enferm`** | 0 | No |
| | 1 | Sí |
| | 2 | Desconocido |
| **`loc_recidiva`** | 1 | Pélvica |
| | 2 | Nodal |
| | 3 | A distancia |
| | 4 | NO CONSTA |
| | 5 | Vaginal |
| | 6 | Peritoneal |
| **`num_recidiva`** | 1 | Única |
| | 2 | Múltiple |
| **`recidiva`** | 0 | No |
| | 1 | Sí |
| | 2 | Desconocido |
| **`Reseccion_macroscopica_complet`**| 0 | No |
| | 1 | Sí |
| **`Tratamiento_RT`** | 0 | No |
| *(Recidiva)* | 1 | Sí |
| **`Tratamiento_sistemico`** | 0 | No |
| *(Recidiva)* | 1 | Sí |
| **`tto_recidiva`** | 0 | No |
| | 1 | Curativo |
| | 2 | Paliativo |
| **`Tt_recidiva_qx`** | 0 | No |
| | 1 | Resección local |
| | 2 | Exanteración |
| | 3 | Otros |
