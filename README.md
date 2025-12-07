# EndoLogic: NSMP Endometrial Risk Stratification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED)
![Status](https://img.shields.io/badge/Status-Hackathon%20MVP-orange)

> **Hack the Uterus! - NEST Challenge**
> *Bridging the gap between molecular biology and clinical decision-making.*

## 🏥 The Problem: The "Gray Zone"
Endometrial cancer is the most common gynecological tumor in developed countries. While molecular classification (TCGA) has revolutionized diagnosis, **50% of patients fall into the NSMP (Non-Specific Molecular Profile) group**.

This group is a **"clinical gray zone"**:
*   It is defined by exclusion (what it is *not*).
*   Prognosis is highly heterogeneous (some cure, some metastasize).
*   Current standard treatment leads to **overtreatment** (unnecessary toxicity) or **undertreatment** (fatal recurrence).

## 💡 Our Solution
**EndoCore** is a Machine Learning-powered Clinical Decision Support System (CDSS) designed specifically for the NSMP subgroup. 

We prioritize a **"Safety-First AI"** approach:
1.  **Risk Stratification:** Classifies patients into Low, Intermediate, or High Risk using clinical variables (LVSI, Grade, Stage) and biomarkers (L1CAM, ER/PR).
2.  **Actionable Insights:** Provides a "Traffic Light" output for immediate clinical interpretation.
3.  **Explainability:** Uses **SHAP (SHapley Additive exPlanations)** to show *why* a specific risk score was assigned, building trust with medical professionals.

## 🚀 Key Features

### 🧠 Advanced ML Engine
*   **Algorithm:** XGBoost optimized for tabular medical data.
*   **Metric Strategy:** We optimize for **F2-Score** rather than Accuracy. We value **Sensitivity (Recall)** twice as much as Precision because missing a cancer recurrence (False Negative) is unacceptable.
*   **Biologically Validated:** The model respects known medical interactions (e.g., *L1CAM+* increases risk, *ER+* decreases risk).

### 🛠 Robust Engineering & DevOps
*   **Dockerized Architecture:** The entire stack (Frontend, Backend, Database) is containerized for reproducibility.
*   **Automated CI/CD:** We utilize **GitHub Actions** for integration and a custom **Webhook** pipeline for continuous deployment to our cloud server.
*   **Database Integration:** **MongoDB** stores patient logs and model inferences for future retraining (Continuous Learning).

## 🏗 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Interactive dashboard for clinicians. |
| **Backend** | Python / Pydantic | Business logic, data validation, and inference. |
| **ML Core** | Scikit-learn / XGBoost | Predictive modeling and SHAP explanation. |
| **Database** | MongoDB | NoSQL storage for logs and feedback. |
| **Infrastructure** | Docker & Compose | Container orchestration. |

## 📂 Project Structure

```text
EndoCore/
├── .github/workflows/      # CI pipelines
├── data/                   # Data storage (gitignored)
├── models/                 # Serialized ML models
├── notebooks/              # Data Science experiments (EDA, Training)
├── src/
│   ├── backend/            # Logic & Database connection
│   ├── frontend/           # Streamlit UI
│   └── utils/              # Helper functions
├── docker-compose.yml      # Service orchestration
├── Dockerfile              # Container definition
└── generate_mock_data.py   # Synthetic data generator for testing