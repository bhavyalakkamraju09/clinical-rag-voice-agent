"""
Directly creates a high-quality golden test set with pre-written general
clinical questions and answers generated from the actual note corpus.
Run: python evaluation/create_golden_set.py
"""
import sys, json, random, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_PATH = Path("evaluation/golden_test_set.json")
NOTES_PATH  = Path("data/raw/clinical_notes.json")

QUESTIONS = [
    "What medications are commonly prescribed for Type 2 Diabetes patients?",
    "What blood pressure readings are documented for hypertensive patients?",
    "What inhalers are prescribed for COPD patients?",
    "What comorbidities appear alongside Congestive Heart Failure?",
    "What HbA1c levels are documented for diabetic patients?",
    "What findings are documented on chest X-rays for heart failure patients?",
    "What smoking histories are documented in patient records?",
    "What follow-up intervals are recommended for chronic disease management?",
    "What anticoagulants are used for atrial fibrillation patients?",
    "What eGFR values are documented for Chronic Kidney Disease patients?",
    "What does an echocardiogram show for heart failure patients?",
    "What specialist referrals are made for cardiac conditions?",
    "What medications are used for hypertension management?",
    "What SpO2 levels are documented for COPD patients?",
    "What Holter monitor findings are documented in patient records?",
    "What renal ultrasound findings appear for CKD patients?",
    "What lipid panel values are documented including LDL and HDL?",
    "What medications are prescribed for osteoporosis management?",
    "What INR values are documented for patients on Warfarin?",
    "What CT chest findings are documented for COPD patients?",
    "What treatment plans are documented for Parkinson's Disease?",
    "What complications are noted in discharge summaries?",
    "What lab values are monitored for patients on anticoagulants?",
    "What medications are prescribed for heart failure management?",
    "What imaging studies are ordered for stroke patients?",
    "What lifestyle modifications are recommended in treatment plans?",
    "What BNP levels are recorded for heart failure patients?",
    "What procedures are documented in procedure notes?",
    "What allergies are commonly documented in patient records?",
    "What medications are prescribed alongside insulin for diabetic patients?",
    "What TSH levels are documented for thyroid patients?",
    "What vital signs are documented for atrial fibrillation patients?",
    "What treatment is documented for Alzheimer's Disease patients?",
    "What are common reasons for hospital admission in these records?",
    "What medications are prescribed for depression management?",
    "What DEXA scan findings are documented for osteoporosis patients?",
    "What exercise habits are mentioned in patient social histories?",
    "What MRI brain findings are documented for neurological patients?",
    "What medications are used for COPD management?",
    "What are the documented treatment plans for anxiety disorders?",
    "What abdominal ultrasound findings are documented?",
    "What calcium and vitamin D doses are prescribed?",
    "What stress test results are documented for cardiac patients?",
    "What medications are prescribed for thyroid conditions?",
    "What troponin levels are documented in urgent visit notes?",
    "What social history details are documented for patients with COPD?",
    "What follow-up plans are documented for post-discharge patients?",
    "What medication doses are adjusted during follow-up visits?",
    "What are the documented vital signs for patients with diabetes?",
    "What imaging is ordered for patients with peripheral artery disease?",
]


def get_answer(question: str, notes: list, generate_fn: callable) -> str:
    sample = random.sample(notes, min(8, len(notes)))
    context = "\n\n---\n\n".join(n["note"][:600] for n in sample)
    prompt = f"""You are a clinical data analyst. Using the clinical notes below, 
answer the question with specific facts and values found in the notes.
Be concise (2-4 sentences). Include actual numbers/values where present.

Clinical notes:
{context}

Question: {question}

Answer:"""
    return generate_fn(prompt, temperature=0.1, max_tokens=200)


def main():
    with open(NOTES_PATH) as f:
        notes = json.load(f)

    if os.getenv("GROQ_API_KEY"):
        from src.llm.groq_client import generate
        print("Using Groq")
    else:
        from src.llm.ollama_client import generate
        print("Using Ollama")

    golden = []
    for i, q in enumerate(QUESTIONS[:50]):
        try:
            answer = get_answer(q, notes, generate)
            golden.append({"question": q, "expected_answer": answer})
            print(f"  [{i+1:02d}] {q[:65]}…")
        except Exception as e:
            print(f"  [skip] {e}")

    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"\n✓ {len(golden)} golden pairs saved → {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
