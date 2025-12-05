# tuning.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "wsd_dataset_semcor.csv")
LEMMA_FILE = os.path.join(BASE_DIR, "top_lemmas_stats.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "tuning")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("[TUNING] Loading data...")
df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=["sense", "lemma"])

# Use only lemmas from your ambiguous list (keeps WSD nature)
lemmas = pd.read_csv(LEMMA_FILE)["lemma"].tolist()
df = df[df["lemma"].isin(lemmas)]

sense_counts = df["sense"].value_counts()
df = df[df["sense"].isin(sense_counts[sense_counts >= 10].index)]

# Build the same combined context used in model scripts
df["combined_context"] = (
    df["context_before_words"].fillna("") + " " +
    df["target_word"].fillna("") + " " +
    df["context_after_words"].fillna("") + " " +
    df["sentence_words"].fillna("")
)

X_text = df["combined_context"]
y = df["sense"]

print("[TUNING] Vectorizing...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=60000
)
X = vectorizer.fit_transform(X_text)

# Basic split to reduce size if needed
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

models = {
    "NB": ComplementNB(),
    "DT": DecisionTreeClassifier(),
    "SVM": LinearSVC(),
    "LR": LogisticRegression(max_iter=4000)
}

param_distributions = {
    "NB": {
        "alpha": [0.1, 0.5, 1.0]
    },
    "DT": {
        "max_depth": [20, 50],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "SVM": {
        "C": [0.1, 0.5, 1.0, 2.0],
        "loss": ["hinge", "squared_hinge"]
    },
    "LR": {
        "C": [0.1, 0.5, 1.0, 2.0],
        "penalty": ["l2"],
        "solver": ["liblinear", "saga"]
    }
}

rows = []

for tag, base_model in models.items():
    print(f"\n[TUNING] Running RandomizedSearchCV for {tag}...")

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions[tag],
        n_iter=5,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score = search.best_score_

    rows.append({
        "model": tag,
        "best_params": best_params,
        "best_macroF1": best_score
    })

    # Save full cv results
    cv_results_df = pd.DataFrame(search.cv_results_)
    cv_results_df.to_csv(
        os.path.join(RESULTS_DIR, f"{tag}_tuning_cv_results.csv"),
        index=False
    )

    print(f"[TUNING] {tag} best_macroF1={best_score:.4f}")
    print(f"[TUNING] {tag} best_params={best_params}")

summary_df = pd.DataFrame(rows)
summary_path = os.path.join(RESULTS_DIR, "tuning_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\n[TUNING] Summary saved to:", summary_path)
