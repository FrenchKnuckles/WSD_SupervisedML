import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, cohen_kappa_score
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(BASE_DIR, "wsd_dataset_semcor.csv")
LEMMA_FILE = os.path.join(BASE_DIR, "top_lemmas_stats.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

lemmas = pd.read_csv(LEMMA_FILE)["lemma"].tolist()

def compute_group_recall(labels, preds, groups):
    mask = labels.isin(groups)
    if mask.sum() == 0:
        return None
    return recall_score(labels[mask], preds[mask], average="macro", zero_division=0)

rows = []

for lemma in lemmas:
    print(f"[SVM] Processing lemma: {lemma}")

    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["sense", "lemma"])

    df_word = df[df["lemma"] == lemma].copy()
    sense_counts = df_word["sense"].value_counts()
    df_word = df_word[df_word["sense"].isin(sense_counts[sense_counts >= 10].index)]

    if df_word["sense"].nunique() < 2:
        print(f"[SVM] Skipping {lemma} (not enough senses)")
        continue

    df_word["combined_context"] = (
        df_word["context_before_words"].fillna("") + " " +
        df_word["target_word"].fillna("") + " " +
        df_word["context_after_words"].fillna("") + " " +
        df_word["sentence_words"].fillna("")
    )

    X = df_word["combined_context"]
    y = df_word["sense"]

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2,
                                 sublinear_tf=True, max_features=60000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LinearSVC(class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    sense_freq = y_train.value_counts()
    rare_senses = sense_freq[sense_freq < 10].index
    medium_senses = sense_freq[(sense_freq >= 10) & (sense_freq < 30)].index
    frequent_senses = sense_freq[sense_freq >= 30].index

    rare_recall = compute_group_recall(y_test, y_pred, rare_senses)
    medium_recall = compute_group_recall(y_test, y_pred, medium_senses)
    frequent_recall = compute_group_recall(y_test, y_pred, frequent_senses)

    train_pred = model.predict(X_train)

    rows.append({
        "lemma": lemma,
        f"SVM_acc": accuracy_score(y_test, y_pred),
        f"SVM_macroF1": f1_score(y_test, y_pred, average="macro"),
        f"SVM_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        f"SVM_kappa": cohen_kappa_score(y_test, y_pred),
        f"SVM_genGap": f1_score(y_train, train_pred, average="macro") -
                                f1_score(y_test, y_pred, average="macro"),
        f"SVM_rareRecall": rare_recall,
        f"SVM_mediumRecall": medium_recall,
        f"SVM_frequentRecall": frequent_recall,
    })

output_path = os.path.join(RESULTS_DIR, "SVM_results.csv")
pd.DataFrame(rows).to_csv(output_path, index=False)
print("SVM results saved to:", output_path)
