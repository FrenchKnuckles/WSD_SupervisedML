import os
import csv
import xml.etree.ElementTree as ET

# ================= CONFIGURATION =================
WINDOW_SIZE = 3
DATA_DIR = "data/semcor"        # input directory
OUTPUT_CSV = "wsd_dataset_semcor.csv"

# CSV header
header = [
    "doc_id", "sent_id", "token_index",
    "target_word", "lemma", "pos", "sense",
    "context_before_words", "context_after_words",
    "sentence_words", "sentence_pos"
]

rows = []


def process_file(filepath):
    """Parse a single XML file and extract training instances."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    sentences = []
    current_sentence = []

    for elem in root.iter("word"):
        token = elem.get("text")
        lemma = elem.get("lemma")
        pos = elem.get("pos")
        sense = elem.get("sense")
        break_level = elem.get("break_level")

        current_sentence.append({
            "token": token,
            "lemma": lemma,
            "pos": pos,
            "sense": sense
        })

        # Check for sentence boundary
        if break_level == "SENTENCE_BREAK":
            sentences.append(current_sentence)
            current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


# ================= MAIN LOOP =================
for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".xml"):
        continue

    xml_path = os.path.join(DATA_DIR, filename)
    print("[Processing]", xml_path)

    sentences = process_file(xml_path)

    for sent_id, tokens in enumerate(sentences):
        for i, tok in enumerate(tokens):
            if tok["sense"] is None:
                continue  # keep only labeled instances

            before = tokens[max(0, i - WINDOW_SIZE): i]
            after = tokens[i + 1: i + 1 + WINDOW_SIZE]

            rows.append([
                filename,                      # doc ID
                sent_id,                       # sentence ID
                i,                             # token index in sentence
                tok["token"],                  # target word
                tok["lemma"],
                tok["pos"],
                tok["sense"],                  # label
                " ".join(t["token"] for t in before),
                " ".join(t["token"] for t in after),
                " ".join(t["token"] for t in tokens),
                " ".join((t["pos"] or "NULL") for t in tokens)
            ])


# ================= SAVE CSV =================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("✔ CSV Created Successfully:", OUTPUT_CSV)
print("Total labeled instances:", len(rows))
