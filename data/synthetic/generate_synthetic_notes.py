"""
Generate 10,000 HIPAA-safe synthetic clinical notes using Faker.
Covers 25 conditions, realistic medications, lab values, imaging,
multi-visit history, specialist referrals, and procedure notes.
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
    "Chronic Kidney Disease Stage 4",
    "Osteoarthritis",
    "Rheumatoid Arthritis",
    "Major Depressive Disorder",
    "Generalized Anxiety Disorder",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Asthma",
    "Coronary Artery Disease",
    "Peripheral Artery Disease",
    "Stroke (Ischemic)",
    "Parkinson's Disease",
    "Alzheimer's Disease",
    "Chronic Liver Disease",
    "Irritable Bowel Syndrome",
    "Gastroesophageal Reflux Disease (GERD)",
    "Anemia (Iron Deficiency)",
    "Obstructive Sleep Apnea",
    "Osteoporosis",
]

MEDICATIONS = [
    "Metformin 500mg twice daily",
    "Metformin 1000mg twice daily",
    "Lisinopril 10mg daily",
    "Lisinopril 20mg daily",
    "Atorvastatin 40mg nightly",
    "Rosuvastatin 20mg nightly",
    "Albuterol inhaler 2 puffs PRN",
    "Furosemide 20mg daily",
    "Furosemide 40mg twice daily",
    "Levothyroxine 50mcg daily",
    "Levothyroxine 100mcg daily",
    "Amlodipine 5mg daily",
    "Amlodipine 10mg daily",
    "Omeprazole 20mg daily",
    "Pantoprazole 40mg daily",
    "Aspirin 81mg daily",
    "Warfarin 5mg daily",
    "Apixaban 5mg twice daily",
    "Carvedilol 12.5mg twice daily",
    "Metoprolol succinate 50mg daily",
    "Sertraline 50mg daily",
    "Escitalopram 10mg daily",
    "Gabapentin 300mg three times daily",
    "Insulin glargine 20 units nightly",
    "Semaglutide 0.5mg weekly",
    "Empagliflozin 10mg daily",
    "Spironolactone 25mg daily",
    "Hydrochlorothiazide 25mg daily",
    "Montelukast 10mg nightly",
    "Tiotropium inhaler 18mcg daily",
    "Budesonide/formoterol 160/4.5mcg twice daily",
    "Donepezil 10mg nightly",
    "Levodopa/carbidopa 25/100mg three times daily",
    "Clopidogrel 75mg daily",
    "Nitroglycerin 0.4mg SL PRN",
    "Colchicine 0.6mg twice daily",
    "Allopurinol 300mg daily",
    "Calcium carbonate 500mg twice daily",
    "Vitamin D3 2000 IU daily",
    "Iron sulfate 325mg daily",
]

SPECIALISTS = [
    "Cardiology", "Nephrology", "Endocrinology", "Pulmonology",
    "Neurology", "Rheumatology", "Gastroenterology", "Hematology",
    "Orthopedics", "Psychiatry", "Sleep Medicine", "Urology",
]

LAB_TEMPLATES = [
    "HbA1c {a1c}%, fasting glucose {gluc} mg/dL, eGFR {egfr} mL/min.",
    "BMP: Na {na}, K {k}, Cr {cr} mg/dL, BUN {bun}. CBC WNL.",
    "TSH {tsh} mIU/L, Free T4 {t4} ng/dL. Lipid panel: LDL {ldl}, HDL {hdl}.",
    "INR {inr}, PT {pt}s. Hgb {hgb} g/dL, platelets {plt}k.",
    "Troponin {trop} ng/mL, BNP {bnp} pg/mL. EKG: {ekg}.",
    "LFTs: ALT {alt}, AST {ast}, total bili {bili}. Albumin {alb} g/dL.",
]

IMAGING_RESULTS = [
    "CXR shows mild cardiomegaly with no acute infiltrates.",
    "Echo: EF 45-50%, mild diastolic dysfunction, no wall motion abnormalities.",
    "CT chest: Hyperinflation consistent with emphysema. No pulmonary embolism.",
    "MRI brain: Scattered white matter changes, no acute infarct.",
    "Renal ultrasound: Bilateral echogenic kidneys, no hydronephrosis.",
    "X-ray knees: Moderate joint space narrowing bilaterally, osteophyte formation.",
    "DEXA scan: T-score -2.6 at lumbar spine, consistent with osteoporosis.",
    "Abdominal ultrasound: Hepatic steatosis, no focal lesions.",
    "Stress test: No inducible ischemia at 85% max heart rate.",
    "Holter monitor: Occasional PACs, no sustained arrhythmia.",
]

SOCIAL_HISTORY = [
    "Never smoker, occasional alcohol use, retired teacher.",
    "Former smoker, 20 pack-year history, quit 5 years ago. Denies alcohol.",
    "Active smoker, 1 PPD x 30 years. Social alcohol use.",
    "Never smoker. Denies alcohol or illicit drug use. Works as engineer.",
    "Former smoker, quit 10 years ago. Occasional alcohol. Sedentary lifestyle.",
    "Never smoker. Moderate alcohol (2-3 drinks/week). Regular exercise.",
    "Active smoker 0.5 PPD. History of alcohol use disorder, currently sober x 2 years.",
]

ALLERGIES = [
    "NKDA",
    "Penicillin (rash)",
    "Sulfa drugs (anaphylaxis)",
    "NSAIDs (GI upset)",
    "Codeine (nausea/vomiting)",
    "Contrast dye (hives) — premedicate before imaging",
    "Lisinopril (angioedema) — avoid ACE inhibitors",
]

NOTE_TYPES = ["follow_up", "new_patient", "procedure", "discharge", "urgent"]


def random_bp(hypertensive=False):
    if hypertensive:
        sys = random.randint(140, 175)
        dia = random.randint(85, 105)
    else:
        sys = random.randint(108, 138)
        dia = random.randint(65, 88)
    return f"{sys}/{dia} mmHg"


def generate_labs():
    template = random.choice(LAB_TEMPLATES)
    return template.format(
        a1c=round(random.uniform(6.5, 11.2), 1),
        gluc=random.randint(95, 280),
        egfr=random.randint(20, 90),
        na=random.randint(136, 145),
        k=round(random.uniform(3.4, 5.2), 1),
        cr=round(random.uniform(0.8, 3.5), 1),
        bun=random.randint(12, 55),
        tsh=round(random.uniform(0.4, 8.5), 2),
        t4=round(random.uniform(0.7, 1.8), 1),
        ldl=random.randint(65, 185),
        hdl=random.randint(32, 72),
        inr=round(random.uniform(1.8, 3.5), 1),
        pt=random.randint(14, 28),
        hgb=round(random.uniform(8.5, 14.5), 1),
        plt=random.randint(120, 380),
        trop=round(random.uniform(0.01, 0.04), 3),
        bnp=random.randint(80, 900),
        ekg=random.choice(["NSR", "sinus tachycardia", "AFib with RVR", "LBBB"]),
        alt=random.randint(18, 95),
        ast=random.randint(15, 88),
        bili=round(random.uniform(0.4, 2.8), 1),
        alb=round(random.uniform(2.8, 4.5), 1),
    )


def generate_note():
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=35, maximum_age=88)
    mrn = fake.numerify("MRN-########")
    age = random.randint(35, 88)
    conditions = random.sample(CONDITIONS, k=random.randint(1, 3))
    primary_dx = conditions[0]
    comorbidities = conditions[1:] if len(conditions) > 1 else []
    meds = random.sample(MEDICATIONS, k=random.randint(3, 7))
    visit_date = fake.date_between(start_date="-2y", end_date="today")
    note_type = random.choice(NOTE_TYPES)
    hypertensive = "Hypertension" in conditions
    specialist = random.choice(SPECIALISTS)
    allergy = random.choice(ALLERGIES)
    social = random.choice(SOCIAL_HISTORY)
    labs = generate_labs() if random.random() > 0.3 else ""
    imaging = random.choice(IMAGING_RESULTS) if random.random() > 0.5 else ""

    vitals = (
        f"BP {random_bp(hypertensive)}, HR {random.randint(56, 108)} bpm, "
        f"RR {random.randint(14, 22)}/min, SpO2 {random.randint(91, 99)}%, "
        f"Temp {round(random.uniform(97.4, 99.4), 1)}°F, "
        f"Wt {random.randint(120, 265)} lbs, BMI {round(random.uniform(18.5, 42.0), 1)}."
    )

    comorbidity_str = (
        f"Comorbidities include {', '.join(comorbidities)}. " if comorbidities else ""
    )

    if note_type == "new_patient":
        hpi = (
            f"{age}-year-old {'male' if random.random()>0.5 else 'female'} presenting as a new patient "
            f"for evaluation of {primary_dx}. {comorbidity_str}"
            f"Patient reports {random.choice(['worsening symptoms over the past 3 months', 'stable symptoms', 'new onset symptoms over the past 4 weeks'])}. "
            f"{'Denies chest pain. ' if random.random()>0.5 else ''}"
            f"{'Reports dyspnea on exertion. ' if random.random()>0.5 else ''}"
            f"{'Endorses fatigue and malaise. ' if random.random()>0.4 else ''}"
        )
    elif note_type == "procedure":
        hpi = (
            f"Patient presenting for {random.choice(['diagnostic colonoscopy', 'cardiac catheterization', 'pulmonary function testing', 'bone marrow biopsy', 'joint injection', 'paracentesis'])}. "
            f"Indication: {primary_dx}. {comorbidity_str}Pre-procedure checklist completed. Informed consent obtained."
        )
    elif note_type == "discharge":
        hpi = (
            f"Patient admitted {random.randint(2, 12)} days ago for {random.choice(['acute exacerbation of', 'decompensated', 'newly diagnosed'])} {primary_dx}. "
            f"{comorbidity_str}Course complicated by {random.choice(['acute kidney injury', 'hospital-acquired pneumonia', 'electrolyte imbalances', 'no major complications'])}. "
            f"Patient clinically improved and medically stable for discharge."
        )
    elif note_type == "urgent":
        hpi = (
            f"Patient presenting urgently with {random.choice(['chest pain', 'shortness of breath', 'altered mental status', 'acute decompensation', 'severe hypertension'])} "
            f"in the setting of known {primary_dx}. {comorbidity_str}"
            f"Onset {random.choice(['2 hours', '4 hours', 'this morning', 'yesterday evening'])} ago."
        )
    else:  # follow_up
        hpi = (
            f"Patient returns for routine follow-up of {primary_dx}. {comorbidity_str}"
            f"Reports {'improvement' if random.random()>0.4 else 'no significant change'} since last visit. "
            f"{'Medication adherence confirmed. ' if random.random()>0.5 else 'Reports occasional missed doses. '}"
            f"{'Denies new symptoms. ' if random.random()>0.5 else ''}"
        )

    plan_options = [
        f"Continue current medications. Reinforce lifestyle modifications. Follow-up in {random.choice([4, 6, 8, 12])} weeks.",
        f"Increase {meds[0].split()[0]} dose. Repeat labs in {random.choice([2, 4])} weeks. Patient educated on signs of toxicity.",
        f"Refer to {specialist} for further evaluation. Continue current regimen. Monitor symptoms closely.",
        f"Add {random.choice(MEDICATIONS).split(' ')[0]} to regimen. Educate patient on new medication side effects.",
        f"Order {random.choice(['echocardiogram', 'renal ultrasound', 'pulmonary function tests', 'HbA1c', 'thyroid panel', 'sleep study'])}. Follow up with results.",
        f"Hospitalization considered but deferred. Strict return precautions given. Follow-up in {random.choice([1, 3, 7])} days.",
    ]
    plan = random.choice(plan_options)

    note_text = (
        f"{'FOLLOW-UP NOTE' if note_type=='follow_up' else 'NEW PATIENT NOTE' if note_type=='new_patient' else 'PROCEDURE NOTE' if note_type=='procedure' else 'DISCHARGE SUMMARY' if note_type=='discharge' else 'URGENT VISIT NOTE'} — {visit_date}\n"
        f"Patient: {name} | DOB: {dob} | Age: {age} | MRN: {mrn}\n"
        f"Allergies: {allergy}\n\n"
        f"Chief Complaint: {note_type.replace('_',' ').title()} for {primary_dx}.\n\n"
        f"History of Present Illness:\n{hpi}\n\n"
        f"Social History: {social}\n\n"
        f"Current Medications:\n"
        + "\n".join(f"  - {m}" for m in meds)
        + f"\n\nVital Signs:\n{vitals}\n\n"
        + (f"Laboratory Results:\n{labs}\n\n" if labs else "")
        + (f"Imaging:\n{imaging}\n\n" if imaging else "")
        + f"Assessment:\n1. {primary_dx} — {'well controlled' if random.random()>0.5 else 'poorly controlled' if random.random()>0.5 else 'stable'}\n"
        + "".join(f"{i+2}. {c}\n" for i, c in enumerate(comorbidities))
        + f"\nPlan:\n{plan}\n\n"
        f"Electronically signed by {fake.name()}, {random.choice(['MD', 'DO', 'NP', 'PA-C'])}"
    )

    return {
        "id": str(uuid.uuid4()),
        "patient_name": name,
        "dob": str(dob),
        "mrn": mrn,
        "age": age,
        "visit_date": str(visit_date),
        "note_type": note_type,
        "primary_condition": primary_dx,
        "comorbidities": comorbidities,
        "medications": meds,
        "note": note_text,
    }


if __name__ == "__main__":
    output_path = Path("data/raw/clinical_notes.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Generating 10,000 synthetic clinical notes...")
    notes = [generate_note() for _ in range(10000)]
    with open(output_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"✓ Generated {len(notes)} notes → {output_path}")
    print(f"  Conditions: {len(CONDITIONS)} unique diagnoses")
    print(f"  Note types: {len(set(n['note_type'] for n in notes))} types")
    print(f"  Medications: {len(MEDICATIONS)} unique drugs")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
