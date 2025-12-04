import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

summary_path = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
strengths_path = os.path.join(RESULTS_DIR, "model_strengths_weaknesses.csv")

if not os.path.exists(summary_path):
    raise FileNotFoundError("Run compare.py first to generate summary CSV.")

STORE_DIR = os.path.join(BASE_DIR, "results/imgs")

summary = pd.read_csv(summary_path)
strengths = pd.read_csv(strengths_path)

summary.set_index("model", inplace=True)


summary[["accuracy","macroF1","recall","kappa"]].plot(
    kind="bar", figsize=(10,6), rot=0, title="Overall Model Performance"
)
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(STORE_DIR, "overall_performance.png"))
plt.close()

summary[["rareRecall","mediumRecall","frequentRecall"]].plot(
    kind="bar", figsize=(10,6), rot=0, title="Recall by Sense Frequency Groups"
)
plt.ylabel("Recall")
plt.tight_layout()
plt.savefig(os.path.join(STORE_DIR, "sense_group_recall.png"))
plt.close()

summary[["genGap"]].plot(kind="bar", figsize=(8,5), rot=0,
                         title="Generalization Gap per Model")
plt.ylabel("Train-Test F1 Difference")
plt.tight_layout()
plt.savefig(os.path.join(STORE_DIR, "generalization_gap.png"))
plt.close()

combined = pd.concat([
    pd.read_csv(os.path.join(RESULTS_DIR, f"{m}_results.csv"))
    for m in ["NB","DT","LR","SVM"] if os.path.exists(os.path.join(RESULTS_DIR, f"{m}_results.csv"))
])

combined.boxplot(column=[col for col in combined.columns if col.endswith("_macroF1")],
                 figsize=(10,6), rot=45)
plt.title("MacroF1 Variance Across Lemmas")
plt.tight_layout()
plt.savefig(os.path.join(STORE_DIR, "macroF1_variance_boxplot.png"))
plt.close()

wins = strengths["model_best"].value_counts()
wins.plot(kind="bar", figsize=(8,5), rot=0, title="Wins Per Lemma")
plt.ylabel("Number of Lemmas Won")
plt.tight_layout()
plt.savefig(os.path.join(STORE_DIR, "wins_per_lemma.png"))
plt.close()

print("All visualizations saved to /results/")
