import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FEATURE_PATH = os.path.join(RESULTS_DIR, "feature_comparison_results.csv")

summary_path = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
strengths_path = os.path.join(RESULTS_DIR, "model_strengths_weaknesses.csv")

if os.path.exists(FEATURE_PATH):
    ablation = pd.read_csv(FEATURE_PATH)
else:
    raise FileNotFoundError("Run feature_compare.py to create feature_comparison_results.csv first.")

if not os.path.exists(summary_path):
    raise FileNotFoundError("Run compare.py first to generate summary CSV.")

STORE_DIR = os.path.join(BASE_DIR, "results/imgs")

summary = pd.read_csv(summary_path)
strengths = pd.read_csv(strengths_path)

summary.set_index("model", inplace=True)
pivot_macro = ablation.pivot_table(values="macroF1", index="feature", columns="model", aggfunc="mean")
pivot_acc = ablation.pivot_table(values="accuracy", index="feature", columns="model", aggfunc="mean")


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

pivot_macro.plot(kind="bar", figsize=(10,6), rot=0,
                 title="MacroF1 Comparison Across Feature Types")
plt.ylabel("Macro F1 Score")
plt.tight_layout()
plt.savefig(os.path.join(STORE_DIR, "feature_macroF1_comparison.png"))
plt.close()

pivot_acc.plot(kind="bar", figsize=(10,6), rot=0,
               title="Accuracy Comparison Across Feature Types")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(STORE_DIR, "feature_accuracy_comparison.png"))
plt.close()

pivot_macro.T.plot(kind="line", marker='o', figsize=(10,6),
                   title="Performance Shift from Window → Sentence → TFIDF+POS")
plt.ylabel("Macro F1 Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(STORE_DIR, "performance_shift_trends.png"))
plt.close()

heatmap_data = ablation.pivot_table(
    values="macroF1",
    index="feature",
    columns="model",
    aggfunc="mean"
)

plt.figure(figsize=(10,7))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={'label': 'Macro F1 Score'}
)

plt.title("Feature vs Model Performance Heatmap (MacroF1)", fontsize=18, fontweight='bold')
plt.xlabel("Model", fontsize=14)
plt.ylabel("Feature Type", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

heatmap_path = os.path.join(STORE_DIR, "feature_model_heatmap.png")
plt.savefig(heatmap_path)
plt.close()


print("All visualizations saved to /results/imgs")