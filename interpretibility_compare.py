import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERP_DIR = os.path.join(BASE_DIR, "models", "results", "interpretability")
STORE_DIR = os.path.join(BASE_DIR, "results", "imgs")
os.makedirs(STORE_DIR, exist_ok=True)

MODEL_TAGS = ["NB", "DT", "LR", "SVM"]

def load_available_lemmas():
    if not os.path.isdir(INTERP_DIR):
        return []

    lemmas = set()
    for fname in os.listdir(INTERP_DIR):
        if "_top_features_" in fname and fname.endswith(".csv"):
            # e.g., NB_top_features_bank.csv
            parts = fname.split("_top_features_")
            if len(parts) == 2:
                lemma = parts[1].replace(".csv", "")
                lemmas.add(lemma)
    return sorted(list(lemmas))

def build_scores_for_lemma(lemma, top_n=20):
    word_scores = defaultdict(lambda: defaultdict(float))

    # Naive Bayes
    nb_path = os.path.join(INTERP_DIR, f"NB_top_features_{lemma}.csv")
    if os.path.exists(nb_path):
        nb_df = pd.read_csv(nb_path)
        for _, row in nb_df.iterrows():
            words = str(row.get("top_likelihood_words", "")).split(",")
            words = [w.strip() for w in words if isinstance(w, str) and w.strip()]
            # Assign scores: top word gets highest
            for rank, w in enumerate(words):
                score = (len(words) - rank)
                word_scores[w]["NB"] += score

    # Logistic Regression
    lr_path = os.path.join(INTERP_DIR, f"LR_top_features_{lemma}.csv")
    if os.path.exists(lr_path):
        lr_df = pd.read_csv(lr_path)
        for _, row in lr_df.iterrows():
            words = str(row.get("top_positive_words", "")).split(",")
            words = [w.strip() for w in words if isinstance(w, str) and w.strip()]
            for rank, w in enumerate(words):
                score = (len(words) - rank)
                word_scores[w]["LR"] += score

    # SVM
    svm_path = os.path.join(INTERP_DIR, f"SVM_top_features_{lemma}.csv")
    if os.path.exists(svm_path):
        svm_df = pd.read_csv(svm_path)
        for _, row in svm_df.iterrows():
            words = str(row.get("top_positive_words", "")).split(",")
            words = [w.strip() for w in words if isinstance(w, str) and w.strip()]
            for rank, w in enumerate(words):
                score = (len(words) - rank)
                word_scores[w]["SVM"] += score

    # Decision Tree
    dt_path = os.path.join(INTERP_DIR, f"DT_top_features_{lemma}.csv")
    if os.path.exists(dt_path):
        dt_df = pd.read_csv(dt_path)
        for _, row in dt_df.iterrows():
            w = str(row.get("feature", "")).strip()
            imp = float(row.get("importance", 0.0))
            if w:
                word_scores[w]["DT"] += imp

    if not word_scores:
        raise SystemExit(f"No interpretability files found for lemma '{lemma}'.")

    # Build DataFrame
    rows = []
    for w, model_dict in word_scores.items():
        row = {"word": w}
        for m in MODEL_TAGS:
            row[m] = model_dict.get(m, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Normalize per model column for nicer visualization
    for m in MODEL_TAGS:
        col_max = df[m].max()
        if col_max > 0:
            df[m] = df[m] / col_max

    # Keep top words by global max score
    df["max_score"] = df[MODEL_TAGS].max(axis=1)
    df = df.sort_values(by="max_score", ascending=False).head(top_n)
    df = df.drop(columns=["max_score"])

    df = df.set_index("word")
    return df

def plot_heatmap(df, lemma):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Normalized Importance Score"}
    )
    plt.title(f"Feature Importance Comparison Across Models (lemma='{lemma}')", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Word Feature", fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(STORE_DIR, f"feature_importance_comparison_{lemma}.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved feature importance comparison to:", out_path)

def main():
    parser = argparse.ArgumentParser(
        description="Compare feature importances across models for a given lemma"
    )
    parser.add_argument(
        "--lemma",
        type=str,
        default=None,
        help="Target lemma for comparison (default: auto-select first available)"
    )
    args = parser.parse_args()

    available = load_available_lemmas()
    if not available:
        raise SystemExit("No interpretability CSVs found in 'models/results/interpretability'.")

    lemma = args.lemma or available[0]
    if lemma not in available:
        print(f"[WARN] Lemma '{lemma}' not found. Available lemmas: {available}")
        raise SystemExit(1)

    print(f"[INTERP] Building feature-importance comparison for lemma: {lemma}")
    df = build_scores_for_lemma(lemma)
    plot_heatmap(df, lemma)

if __name__ == "__main__":
    main()
