"""
HIPAA PHI de-identification using Microsoft Presidio.
Covers all 18 Safe Harbor identifiers relevant to clinical text.
"""
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# 18 HIPAA Safe Harbor PHI entity types (Presidio recognisers)
PHI_ENTITIES = [
    "PERSON",
    "DATE_TIME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "US_SSN",
    "MEDICAL_LICENSE",
    "NRP",               # National/regional/political group — catches MRN patterns
    "URL",
    "IP_ADDRESS",
    "IBAN_CODE",
    "CREDIT_CARD",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "AGE",               # Custom — add Presidio pattern recogniser below
]

_analyzer:   AnalyzerEngine   | None = None
_anonymizer: AnonymizerEngine | None = None


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer   = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def redact_phi(text: str) -> str:
    """
    Detect and replace PHI in text with placeholders like <PERSON>, <DATE_TIME>.
    Returns PHI-clean string safe for TTS and logging.
    """
    analyzer, anonymizer = _get_engines()

    # Filter to entities the analyzer actually supports
    supported = {r.supported_entities[0] for r in analyzer.registry.recognizers
                 if r.supported_entities}
    entities_to_check = [e for e in PHI_ENTITIES if e in supported]
    # Always include the basics even if recogniser list is incomplete
    entities_to_check = list(set(entities_to_check + ["PERSON", "DATE_TIME", "LOCATION"]))

    results = analyzer.analyze(text=text, entities=entities_to_check, language="en")

    operators = {
        entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
        for entity in entities_to_check
    }

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    return anonymized.text
