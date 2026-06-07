"""
Generate 50 diverse golden QA pairs from synthetic clinical notes for RAGAS eval.
Run: python evaluation/generate_golden_set.py
"""
import sys, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_PATH = Path("evaluation/golden_test_set.json")
NOTES_PATH  = Path("data/raw/clinical_notes.json")

QUESTION_TYPES = [
    "medications",
    "vital signs",
    "diagnosis and conditions",
    "lab results",
    "treatment plan",
    "social history",
    "imaging results",
]

GENERATION_PROMPT = """You are generating evaluation questions for a clinical RAG system.

Given this clinical note, generate ONE specific factual question of type: {qtype}

The question must:
- Be answerable DIRECTLY from the note text
- Ask about {qtype} specifically
- NOT ask about the patient's name or age
- Be clinically meaningful

Clinical note:
{note}

Respond in this exact JSON format with no other text:
{{"question": "...", "expected_answer": "..."}}"""


def main():
    with open(NOTES_PATH) as f:
        notes = json.load(f)

    import os
    if os.getenv("GROQ_API_KEY"):
        from src.llm.groq_client import generate
    else:
        from src.llm.ollama_client import generate

    # Filter notes that have rich content
    rich_notes = [n for n in notes if len(n["note"]) > 600]
    sample = random.sample(rich_notes, min(80, len(rich_notes)))

    golden = []
    qtype_cycle = QUESTION_TYPES * 10  # cycle through types

    for i, note in enumerate(sample):
        if len(golden) >= 50:
            break
        qtype = qtype_cycle[i % len(qtype_cycle)]
        prompt = GENERATION_PROMPT.format(
            qtype=qtype,
            note=note["note"][:1000],
        )
        try:
            resp = generate(prompt, temperature=0.4)
            resp = resp.strip().strip("```json").strip("```").strip()
            # Find JSON in response
            start = resp.find("{")
            end = resp.rfind("}") + 1
            if start >= 0 and end > start:
                item = json.loads(resp[start:end])
                if ("question" in item and "expected_answer" in item
                        and "age" not in item["question"].lower()
                        and "name" not in item["question"].lower()
                        and len(item["expected_answer"]) > 10):
                    golden.append(item)
                    print(f"  [{len(golden):02d}] [{qtype}] {item['question'][:65]}…")
        except Exception as e:
            print(f"  [skip] {e}")

    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"\n✓ {len(golden)} diverse golden pairs saved → {GOLDEN_PATH}")

    # Print type distribution
    print("\nQuestion type distribution:")
    from collections import Counter
    # Re-derive types from order
    for qtype in QUESTION_TYPES:
        print(f"  {qtype}: ~{len(golden)//len(QUESTION_TYPES)}")


if __name__ == "__main__":
    main()
