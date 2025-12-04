import os
import pandas as pd
from tabulate import tabulate
# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

model_tags = ["NB", "DT", "LR", "SVM"]

results = []

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
        f"{tag}_recall": "recall",
        f"{tag}_kappa": "kappa",
        f"{tag}_genGap": "genGap",
        f"{tag}_rareRecall": "rareRecall",
        f"{tag}_mediumRecall": "mediumRecall",
        f"{tag}_frequentRecall": "frequentRecall"
    }
    df = df.rename(columns=rename_map)
    df["model"] = tag
    results.append(df)

if not results:
    raise SystemExit("No result files found. Run the model scripts first.")

combined = pd.concat(results, ignore_index=True)

metrics_cols = ["accuracy","macroF1","recall","kappa","genGap",
                "rareRecall","mediumRecall","frequentRecall"]

summary = combined.groupby("model")[metrics_cols].mean().sort_values(by="macroF1", ascending=False)
summary.to_csv(os.path.join(RESULTS_DIR, "model_comparison_summary.csv"))
print("Summary saved.")

best = combined.loc[combined.groupby("lemma")["macroF1"].idxmax()]
worst = combined.loc[combined.groupby("lemma")["macroF1"].idxmin()]
analysis = best.merge(worst, on="lemma", suffixes=("_best","_worst"))
analysis.to_csv(os.path.join(RESULTS_DIR, "model_strengths_weaknesses.csv"), index=False)

print("\n===== Overall Model Performance =====")
print(tabulate(summary, headers='keys', tablefmt='psql'))