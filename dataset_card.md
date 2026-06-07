---
license: mit
task_categories:
- question-answering
- text-retrieval
language:
- en
tags:
- clinical
- medical
- rag
- hipaa
- synthetic
- healthcare-ai
size_categories:
- 10K<n<100K
---

# Clinical RAG Synthetic Dataset

## Dataset Summary

10,000 HIPAA-safe synthetic clinical notes generated for training and evaluating
clinical RAG (Retrieval-Augmented Generation) systems. All patient data is
completely synthetic — generated using Faker — and contains no real patient information.

## Dataset Details

| Property | Value |
|----------|-------|
| Total notes | 10,000 |
| Unique diagnoses | 25 |
| Note types | 5 (follow-up, new patient, procedure, discharge, urgent) |
| Unique medications | 40 |
| Lab panel types | 6 |
| Imaging report types | 10 |
| Avg note length | ~650 tokens |
| Language | English |

## Diagnoses Covered

Type 2 Diabetes Mellitus, Hypertension, COPD, Congestive Heart Failure,
Atrial Fibrillation, Chronic Kidney Disease (Stage 3 & 4), Osteoarthritis,
Rheumatoid Arthritis, Major Depressive Disorder, Generalized Anxiety Disorder,
Hypothyroidism, Hyperthyroidism, Asthma, Coronary Artery Disease,
Peripheral Artery Disease, Ischemic Stroke, Parkinson's Disease,
Alzheimer's Disease, Chronic Liver Disease, IBS, GERD,
Iron Deficiency Anemia, Obstructive Sleep Apnea, Osteoporosis

## Note Structure

Each note includes:
- Patient demographics (synthetic)
- Chief complaint
- History of present illness
- Social history (smoking, alcohol, exercise)
- Current medications (2-7 per note)
- Vital signs (BP, HR, RR, SpO2, Temp, BMI)
- Lab results (HbA1c, BMP, CBC, lipids, INR, BNP, troponin, LFTs)
- Imaging reports (CXR, Echo, CT, MRI, ultrasound, DEXA)
- Assessment and plan

## Intended Use

- Developing and benchmarking clinical RAG systems
- Training clinical NLP models
- Testing PHI de-identification pipelines
- Evaluating medical question-answering systems

## How to Generate

```python
# Clone the repository
git clone https://github.com/bhavyalakkamraju09/clinical-rag-voice-agent
cd clinical-rag-voice-agent
pip install faker
python data/synthetic/generate_synthetic_notes.py
```

## Privacy & Safety

All data is 100% synthetic. No real patient information was used.
The dataset is designed to mimic real clinical notes for AI development
purposes only and should not be used for medical decision-making.

## Citation

```
@misc{lakkamraju2026clinicalrag,
  author = {Lakkamraju, Bhavyasri},
  title = {Synthetic Clinical Notes Dataset for RAG Evaluation},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/bhavyalakkamraju09/synthetic-clinical-notes}
}
```

## Related Project

Built for the [Clinical RAG Voice Agent](https://github.com/bhavyalakkamraju09/clinical-rag-voice-agent) —
a HIPAA-compliant multi-turn voice QA system with hybrid retrieval and PHI redaction.

[Live Demo](https://huggingface.co/spaces/bhavyalakkamraju09/clinical-rag-voice-agent)
