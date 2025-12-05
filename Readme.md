# Supervised Word Sense Disambiguation on SemCor
This repository implements a complete Word Sense Disambiguation (WSD) pipeline using **classical supervised machine learning models**.  
The system evaluates four algorithms — **Linear SVM, Logistic Regression, Complement Naive Bayes, and Decision Trees** — on the **SemCor corpus** (WordNet-sense-annotated) and performs **feature ablation** and **interpretability analysis**.

---

## Running the Pipeline

### Run the full pipeline
```bash
python run_all.py
```
Or run the individual modules:
```bash
python dataset_convert.py
python ambiguous_lemma_extract.py
python models/SVM.py
python feature_compare.py
python graphical_compare.py
python interpretibility_compare.py --lemma accept
```

---
## Project Overview

### **Research Question**
How effectively can classical supervised learning algorithms (SVM, NB, DT, LR) distinguish between multiple senses of the same word in context, and which model performs best on SemCor-based WSD?

### **Key Contributions**
- Full data-processing pipeline from SemCor XML → structured CSV
- Independent per-lemma multiclass classification
- Comparative evaluation across four ML models
- Ablation study of feature engineering strategies
- Interpretability heatmaps showing context words that drive predictions
- Hyperparameter tuning and generalisation analysis


## Methodology Summary
The pipeline includes:
1. **Dataset conversion** from SemCor XML to structured CSV
2. **Ambiguous lemma extraction** (`ambiguous_lemma_extract.py`)
3. **Per-lemma model training** (NB, DT, LR, SVM)
4. **Model comparison & evaluation** (`compare.py`)
5. **Feature ablation experiments** (`feature_compare.py`)
6. **Hyperparameter tuning** using RandomizedSearchCV
7. **Graph generation** (`graphical_compare.py`)
8. **Interpretability analysis** (`interpretibility_compare.py`)

Evaluation metrics include:
- Accuracy
- Macro-F1, Macro Recall
- Cohen’s Kappa
- Generalisation Gap
- Recall on Rare / Medium / Frequent senses

---

## Results Summary

| Model | Accuracy | Macro-F1 | Generalisation Gap |
|--------|----------|----------|--------------------|
| **SVM** | 0.7899 | **0.671** | 0.310 |
| **LR** | 0.7887 | **0.670** | 0.302 |
| NB | 0.7656 | 0.636 | **0.291** |
| DT | 0.7003 | 0.579 | **0.391** |

### Feature Ablation
- **TF-IDF Window + POS** gives best performance
- Sentence-level features perform worst and cause high noise

### Visual Outputs
All graphs are stored in `/results/imgs/` including:
- `overall_performance.png`
- `sense_group_recall.png`
- `generalization_gap.png`
- `feature_model_heatmap.png`
- `macroF1_variance_boxplot.png`
- `feature_importance_comparison_accept.png` (interpretability)

---

## Installation

```bash
git clone https://github.com/<yourusername>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
