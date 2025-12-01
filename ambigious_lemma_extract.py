import pandas as pd

CSV_FILE = "wsd_dataset_semcor.csv"
OUTPUT_FILE = "top_lemmas_stats.csv"
MIN_SAMPLES_PER_LEMMA = 50       # ignore tiny lemmas
MIN_SENSE_COUNT = 2              # must be ambiguous

print("Loading CSV")
df = pd.read_csv(CSV_FILE)

# Safety: drop missing lemmas or senses
df = df.dropna(subset=["lemma", "sense"])

# Group By lemma
lemma_stats = df.groupby("lemma").agg(
    total_samples=("sense", "count"),
    num_senses=("sense", "nunique")
).reset_index()

# only takes ambigious lemmas
lemma_stats = lemma_stats[
    (lemma_stats["num_senses"] >= MIN_SENSE_COUNT) &
    (lemma_stats["total_samples"] >= MIN_SAMPLES_PER_LEMMA)
]

# sorts by most sensees and samples
lemma_stats = lemma_stats.sort_values(
    by=["num_senses", "total_samples"],
    ascending=[False, False]
)


lemma_stats.to_csv(OUTPUT_FILE, index=False)
print("Top ambiguous lemma list saved:", OUTPUT_FILE)
print("\nTop 30 Lemmas Preview")
print(lemma_stats.head(30))
