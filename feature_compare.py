import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, cohen_kappa_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "wsd_dataset_semcor.csv")
RESULTS_FILE = os.path.join(BASE_DIR, "results", "feature_ablation_results.csv")

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=["sense", "lemma"])

sense_counts = df["sense"].value_counts()
df = df[df["sense"].isin(sense_counts[sense_counts >= 10].index)]

def build_window_features(df):
    return (
        df["context_before_words"].fillna("") + " " +
        df["target_word"].fillna("") + " " +
        df["context_after_words"].fillna("")
    )

def build_sentence_features(df):
    return df["sentence_words"].fillna("")

def build_pos_augmented(df):
    return (
        df["context_before_words"].fillna("") + " " +
        df["target_word"].fillna("") + " " +
        df["context_after_words"].fillna("") + " " +
        df["sentence_pos"].fillna("")
    )

FEATURES = {
    "window": build_window_features,
    "sentence": build_sentence_features,
    "tfidf_pos": build_pos_augmented
}

MODELS = {
    "NB": ComplementNB(alpha=0.5),
    "SVM": LinearSVC(class_weight="balanced"),
    "LR": LogisticRegression(max_iter=500, solver="saga", penalty="l2"),
    "DT": DecisionTreeClassifier(max_depth=50)
}


rows = []

for feat_name, feat_func in FEATURES.items():
    print(f"\n=== Running feature experiment: {feat_name} ===")

    X = feat_func(df)
    y = df["sense"]

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,1),
                                 min_df=3, sublinear_tf=True, max_features=60000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    for tag, model in MODELS.items():
        print(f" -> Fitting {tag} ...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        train_pred = model.predict(X_train)

        rows.append({
            "model": tag,
            "feature": feat_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "macroF1": f1_score(y_test, y_pred, average="macro"),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(y_test, y_pred),
            "genGap": f1_score(y_train, train_pred, average="macro") -
                      f1_score(y_test, y_pred, average="macro")
        })

pd.DataFrame(rows).to_csv(RESULTS_FILE, index=False)
print("\nFeature ablation results saved to:", RESULTS_FILE)
