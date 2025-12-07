# Project Context: Hack the Uterus! - NEST Challenge

## 1. Project Overview
**Project Name:** NEST (NSMP Endometrial Stratification Tool)
**Type:** Medical Hackathon (Hack the Uterus!)
**Goal:** Develop a Machine Learning model and a Clinical Decision Support System (CDSS) to predict the risk of recurrence in patients with **NSMP (Non-Specific Molecular Profile)** Endometrial Cancer.
**Target Audience:** Gynecologic Oncologists. The tool must be interpretable, actionable ("Traffic Light" risk system), and integrated into a clinical workflow.

## 2. Medical Domain Knowledge (CRITICAL)
The core problem is the **NSMP subgroup**, which represents 50% of endometrial cancer cases. It is a "catch-all" group defined by exclusion (Not POLE, Not MMRd, Not p53-abn).
*   **The "Gray Zone":** Prognosis varies widely. We need to split this group into Low, Intermediate, and High Risk.
*   **"Safety First" Approach:** The cost of a False Negative (missing a recurrence) is fatal. The cost of a False Positive is overtreatment (toxicity). We prioritize **Sensitivity (Recall)**.

## 3. Technical Architecture
We are building a **Mock MVP** because real data is not yet available. We need a robust architecture ready for deployment.

### Tech Stack
*   **Language:** Python 3.9+
*   **Frontend:** Streamlit (Web Dashboard).
*   **Backend:** Python logic (Pydantic for validation, Scikit-learn/XGBoost for inference).
*   **Database:** MongoDB (NoSQL). Used to store patient input logs and potential "human-in-the-loop" feedback.
*   **Infrastructure:** Docker & Docker Compose.

### Directory Structure
??

### Instructions for the AI

When asked to help with an .ipynb file, please do not try to edit it directly, just give the code.