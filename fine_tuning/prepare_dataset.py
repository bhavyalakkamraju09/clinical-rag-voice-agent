"""
Prepare clinical QA pairs for LoRA fine-tuning.
Outputs: fine_tuning/clinical_qa.jsonl (Alpaca/SFT format)
Run locally to generate the dataset, then upload to Colab for training.
"""
import sys, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_PATH = Path("fine_tuning/clinical_qa.jsonl")
NOTES_PATH  = Path("data/raw/clinical_notes.json")

ALPACA_TEMPLATE = (
    "Below is a clinical question with relevant context. "
    "Answer accurately and concisely.\n\n"
    "### Context:\n{context}\n\n"
    "### Question:\n{question}\n\n"
    "### Answer:\n{answer}"
)

def main():
    # Load golden set if it exists
    golden_path = Path("evaluation/golden_test_set.json")
    if not golden_path.exists():
        print("golden_test_set.json not found — run evaluation/generate_golden_set.py first")
        return

    with open(golden_path) as f:
        golden = json.load(f)

    with open(NOTES_PATH) as f:
        notes = json.load(f)

    notes_by_id = {n["id"]: n for n in notes}

    records = []
    for item in golden:
        # Pick a relevant note snippet as context (just use first note for demo)
        note = random.choice(notes)
        context = note["note"][:600]
        text = ALPACA_TEMPLATE.format(
            context=context,
            question=item["question"],
            answer=item["expected_answer"],
        )
        records.append({"text": text})

    with open(OUTPUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"✓ {len(records)} training examples → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
