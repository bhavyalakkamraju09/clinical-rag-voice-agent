"""
Generate 500 HIPAA-safe synthetic clinical notes using Faker.
Real patient data is never used — all names/DOBs are fake.
Run: python data/synthetic/generate_synthetic_notes.py
"""
import random
import json
import uuid
from pathlib import Path
from faker import Faker

fake = Faker()
random.seed(42)

CONDITIONS = [
    "Type 2 Diabetes Mellitus",
    "Hypertension",
    "Chronic Obstructive Pulmonary Disease (COPD)",
    "Congestive Heart Failure (CHF)",
    "Atrial Fibrillation",
    "Chronic Kidney Disease Stage 3",
    "Osteoarthritis",
    "Major Depressive Disorder",
    "Hypothyroidism",
    "Asthma",
]

MEDICATIONS = [
    "Metformin 500mg twice daily",
    "Lisinopril 10mg daily",
    "Atorvastatin 40mg nightly",
    "Albuterol inhaler PRN",
    "Furosemide 20mg daily",
    "Levothyroxine 50mcg daily",
    "Amlodipine 5mg daily",
    "Omeprazole 20mg daily",
    "Aspirin 81mg daily",
    "Warfarin 5mg daily",
    "Carvedilol 12.5mg twice daily",
    "Sertraline 50mg daily",
]

VITALS_TEMPLATES = [
    "BP {bp}, HR {hr} bpm, RR {rr}/min, SpO2 {spo2}%, Temp {temp}°F.",
    "Vital signs: BP {bp}, pulse {hr}, respiratory rate {rr}, oxygen saturation {spo2}%.",
]

PLAN_TEMPLATES = [
    "Continue current medications. Follow-up in {weeks} weeks.",
    "Adjust {med} dose. Repeat labs in {days} days. Patient educated on diet.",
    "Refer to {specialist}. Continue current regimen. Monitor symptoms.",
]

SPECIALISTS = ["Cardiology", "Nephrology", "Endocrinology", "Pulmonology", "Neurology"]


def random_bp():
    sys = random.randint(110, 160)
    dia = random.randint(70, 100)
    return f"{sys}/{dia} mmHg"


def generate_note():
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=40, maximum_age=85)
    mrn = fake.numerify("MRN-########")
    condition = random.choice(CONDITIONS)
    meds = random.sample(MEDICATIONS, k=random.randint(2, 4))
    visit_date = fake.date_between(start_date="-1y", end_date="today")

    vitals = random.choice(VITALS_TEMPLATES).format(
        bp=random_bp(),
        hr=random.randint(58, 105),
        rr=random.randint(14, 22),
        spo2=random.randint(94, 99),
        temp=round(random.uniform(97.6, 99.2), 1),
    )

    plan = random.choice(PLAN_TEMPLATES).format(
        weeks=random.choice([2, 4, 6, 8]),
        days=random.choice([7, 14, 30]),
        med=random.choice(meds).split()[0],
        specialist=random.choice(SPECIALISTS),
    )

    note_text = (
        f"CLINICAL NOTE — {visit_date}\n"
        f"Patient: {name} | DOB: {dob} | MRN: {mrn}\n\n"
        f"Chief Complaint: Follow-up visit for {condition}.\n\n"
        f"History of Present Illness:\n"
        f"Patient presents for routine follow-up of {condition}. "
        f"Reports {'improvement' if random.random() > 0.4 else 'no significant change'} "
        f"since last visit. "
        f"{'Denies chest pain or shortness of breath. ' if random.random() > 0.5 else ''}"
        f"{'Reports occasional fatigue. ' if random.random() > 0.5 else ''}\n\n"
        f"Current Medications:\n"
        + "\n".join(f"  - {m}" for m in meds)
        + f"\n\nVital Signs:\n{vitals}\n\n"
        f"Assessment & Plan:\n{condition} — {plan}\n\n"
        f"Electronically signed by {fake.name()}, MD"
    )

    return {
        "id": str(uuid.uuid4()),
        "patient_name": name,  # will be redacted by Presidio before any output
        "dob": str(dob),
        "mrn": mrn,
        "visit_date": str(visit_date),
        "condition": condition,
        "medications": meds,
        "note": note_text,
    }


if __name__ == "__main__":
    output_path = Path("data/raw/clinical_notes.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes = [generate_note() for _ in range(500)]
    with open(output_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"✓ Generated {len(notes)} synthetic clinical notes → {output_path}")
