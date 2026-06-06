"""
Generate 50 golden QA pairs from synthetic clinical notes for RAGAS eval.
Uses the LLM itself to generate Q+A pairs — quick bootstrapping approach.
Run: python evaluation/generate_golden_set.py
"""
import sys, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_PATH = Path("evaluation/golden_test_set.json")
NOTES_PATH  = Path("data/raw/clinical_notes.json")


GENERATION_PROMPT = """Given this clinical note, generate ONE factual question and its correct answer.
The question should be answerable directly from the note.

Clinical note:
{note}

Respond in this exact JSON format:
{{"question": "...", "expected_answer": "..."}}

JSON only, no other text."""


def main():
    with open(NOTES_PATH) as f:
        notes = json.load(f)

    import os
    if os.getenv("GROQ_API_KEY"):
        from src.llm.groq_client import generate
    else:
        from src.llm.ollama_client import generate

    sample = random.sample(notes, min(60, len(notes)))
    golden = []

    for note in sample:
        if len(golden) >= 50:
            break
        prompt = GENERATION_PROMPT.format(note=note["note"][:800])
        try:
            resp = generate(prompt, temperature=0.3)
            # Strip markdown fences if present
            resp = resp.strip().strip("```json").strip("```").strip()
            item = json.loads(resp)
            if "question" in item and "expected_answer" in item:
                golden.append(item)
                print(f"  [{len(golden):02d}] {item['question'][:70]}…")
        except Exception as e:
            print(f"  [skip] {e}")

    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"\n✓ {len(golden)} golden pairs saved → {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
