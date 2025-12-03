import os
import pandas as pd

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

model_tags = ["NB", "DT", "LR", "SVM"]

combined_list = []

for tag in model_tags:
    path = os.path.join(RESULTS_DIR, f"{tag}_results.csv")
    if not os.path.exists(path):
        print(f"[!] Missing results file for {tag}: {path} (skipping)")
        continue

    df = pd.read_csv(path)

    # Rename model-specific columns to common names
    rename_map = {
        f"{tag}_acc": "accuracy",
        f"{tag}_macroF1": "macroF1",
        f"{tag}_precision": "precision",
        f"{tag}_recall": "recall",
    }
    df = df.rename(columns=rename_map)
    df["model"] = tag

    # Keep only needed cols
    df = df[["lemma", "model", "accuracy", "macroF1", "precision", "recall"]]
    combined_list.append(df)

if not combined_list:
    raise SystemExit("No result files found. Run the model scripts first.")

combined = pd.concat(combined_list, ignore_index=True)

# ===== Overall summary per model =====
summary = combined.groupby("model").agg(
    accuracy=("accuracy", "mean"),
    macroF1=("macroF1", "mean"),
    precision=("precision", "mean"),
    recall=("recall", "mean"),
).sort_values(by="macroF1", ascending=False)

summary_path = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
summary.to_csv(summary_path)
print("✔ Saved overall metric summary to", summary_path)

# ===== Best & worst model per lemma (based on macroF1) =====
best_per_lemma = combined.loc[combined.groupby("lemma")["macroF1"].idxmax()]
best_per_lemma = best_per_lemma.rename(columns={
    "model": "best_model",
    "macroF1": "best_macroF1",
    "accuracy": "best_accuracy",
    "precision": "best_precision",
    "recall": "best_recall",
})

worst_per_lemma = combined.loc[combined.groupby("lemma")["macroF1"].idxmin()]
worst_per_lemma = worst_per_lemma.rename(columns={
    "model": "worst_model",
    "macroF1": "worst_macroF1",
    "accuracy": "worst_accuracy",
    "precision": "worst_precision",
    "recall": "worst_recall",
})

analysis = best_per_lemma.merge(worst_per_lemma, on="lemma")
analysis_path = os.path.join(RESULTS_DIR, "model_strengths_weaknesses.csv")
analysis.to_csv(analysis_path, index=False)
print("✔ Saved strengths & weaknesses to", analysis_path)

# ===== Console summaries =====
print("\n===== Overall Model Performance (mean over lemmas) =====")
print(summary)

print("\n===== Where each model excels (top 5 lemmas by macroF1) =====")
for tag in summary.index:
    top_cases = analysis[analysis["best_model"] == tag].sort_values(
        by="best_macroF1", ascending=False
    ).head(5)
    if not top_cases.empty:
        print(f"\n{tag} best cases:")
        print(top_cases[["lemma", "best_macroF1", "best_accuracy", "best_precision", "best_recall"]])

print("\n===== Where each model is weakest (bottom 5 lemmas by macroF1) =====")
for tag in summary.index:
    weak_cases = analysis[analysis["worst_model"] == tag].sort_values(
        by="worst_macroF1", ascending=True
    ).head(5)
    if not weak_cases.empty:
        print(f"\n{tag} weakest cases:")
        print(weak_cases[["lemma", "worst_macroF1", "worst_accuracy", "worst_precision", "worst_recall"]])
