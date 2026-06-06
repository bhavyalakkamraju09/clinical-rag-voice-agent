"""
Semantic chunking for clinical notes.
Uses RecursiveCharacterTextSplitter with clinical-aware separators.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Clinical notes have natural section breaks — prioritise those
CLINICAL_SEPARATORS = [
    "\n\nASSESSMENT",
    "\n\nHISTORY",
    "\n\nCHIEF COMPLAINT",
    "\n\nMEDICATIONS",
    "\n\nVITAL",
    "\n\n",
    "\n",
    " ",
]


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CLINICAL_SEPARATORS,
    )


def chunk_notes(notes: list[dict]) -> list[dict]:
    """
    Takes raw note dicts, returns flat list of chunk dicts.
    Each chunk carries metadata: doc_id, condition, visit_date.
    """
    splitter = get_splitter()
    chunks = []
    for note in notes:
        splits = splitter.split_text(note["note"])
        for i, text in enumerate(splits):
            chunks.append({
                "text": text,
                "doc_id": note["id"],
                "condition": note["primary_condition"],
                "visit_date": note.get("visit_date", ""),
                "chunk_index": i,
            })
    return chunks
