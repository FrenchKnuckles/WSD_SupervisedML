import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB

# Load dataset
df = pd.read_csv("wsd_dataset_semcor.csv")

# Remove very rare senses (< 5 samples)
sense_counts = df["sense"].value_counts()
df = df[df["sense"].isin(sense_counts[sense_counts >= 5].index)]

print("Remaining instances:", len(df))
print("Remaining senses:", df["sense"].nunique())

# Build improved context feature (avoid using entire sentence)
df["combined_context"] = (
    df["context_before_words"].fillna("") + " " +
    df["target_word"].fillna("") + " " +
    df["context_after_words"].fillna("")
)

X = df["combined_context"]
y = df["sense"]

# Vectorizer tuned for NB
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,              # remove very rare words
    sublinear_tf=True,      # log scaling to reduce huge frequency differences
    max_features=80000
)

X_vec = vectorizer.fit_transform(X)

# Train-test split (still stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Complement Naive Bayes (better for imbalance)
model = ComplementNB(alpha=0.5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Improved Naive Bayes Results (ComplementNB) ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 5), "%")
print("Macro F1:", round(f1_score(y_test, y_pred, average="macro"), 5))

print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
#print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
