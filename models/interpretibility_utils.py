import numpy as np
import pandas as pd
import os

RESULTS_INTERPRET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "interpretability")
os.makedirs(RESULTS_INTERPRET, exist_ok=True)


# -------- Logistic Regression & SVM --------
def extract_linear_features(model, vectorizer, lemma, model_tag, top_n=20):
    if len(model.classes_) < 2:
        print(f"[{model_tag}] Skipping interpretability for {lemma} (single-class model)")
        return

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_

    rows = []

    # Binary classification case
    if len(model.classes_) == 2:
        top_pos = feature_names[np.argsort(coefs[0])[-top_n:]][::-1]
        top_neg = feature_names[np.argsort(coefs[0])[:top_n]]

        rows.append({
            "sense": model.classes_[0],
            "top_positive_words": ", ".join(top_pos)
        })
        rows.append({
            "sense": model.classes_[1],
            "top_negative_words": ", ".join(top_neg)
        })

    # Multi-class (one-vs-rest)
    else:
        for idx, cls in enumerate(model.classes_):
            top_pos = feature_names[np.argsort(coefs[idx])[-top_n:]][::-1]
            top_neg = feature_names[np.argsort(coefs[idx])[:top_n]]

            rows.append({
                "sense": cls,
                "top_positive_words": ", ".join(top_pos),
                "top_negative_words": ", ".join(top_neg),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_INTERPRET, f"{model_tag}_top_features_{lemma}.csv"), index=False)
    print(f"[{model_tag}] Interpretability saved for {lemma}")


# -------- Naive Bayes --------
def extract_nb_features(model, vectorizer, lemma, model_tag, top_n=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    log_probs = model.feature_log_prob_

    rows = []
    for idx, cls in enumerate(model.classes_):
        top_words = feature_names[np.argsort(log_probs[idx])[-top_n:]][::-1]
        bottom_words = feature_names[np.argsort(log_probs[idx])[:top_n]]
        rows.append({
            "sense": cls,
            "top_likelihood_words": ", ".join(top_words),
            "low_likelihood_words": ", ".join(bottom_words)
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_INTERPRET, f"{model_tag}_top_features_{lemma}.csv"), index=False)
    print(f"[{model_tag}] Interpretability saved for {lemma}")


# -------- Decision Tree --------
def extract_dt_features(model, vectorizer, lemma, model_tag, top_n=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    importances = model.feature_importances_

    top_idx = np.argsort(importances)[::-1][:top_n]
    top_features = feature_names[top_idx]
    scores = importances[top_idx]

    df = pd.DataFrame({
        "feature": top_features,
        "importance": scores
    })

    df.to_csv(os.path.join(RESULTS_INTERPRET, f"{model_tag}_top_features_{lemma}.csv"), index=False)
    print(f"[{model_tag}] Interpretability saved for {lemma}")
