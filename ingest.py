"""
ingest.py - Batch PDF to ChromaDB ingestion pipeline for GemmaGenius
=====================================================================
Reads a PDF, chunks it into paragraph blocks, sends each chunk to a
local Ollama model (gemma4:e4b) to generate structured concept JSON,
embeds each concept with nomic-embed-text, and upserts it into a local
ChromaDB vector database.

Usage:
    python ingest.py <path/to/file.pdf> [options]

Options:
    --dry-run            Print generated JSON to terminal; do NOT embed or write.
    --model NAME         Ollama generation model (default: gemma4:e4b).
    --embed-model NAME   Ollama embedding model (default: nomic-embed-text).
    --chroma-path PATH   Directory for the ChromaDB store (default: ./chroma_db).
    --collection NAME    ChromaDB collection name (default: curriculum).
    --max-chars N        Max characters per chunk (default: 3000).
    --chunk-limit N      Process at most N chunks (default: all).
    --subject NAME       Subject category tag (default: General).

Examples:
    python ingest.py textbook_chapter.pdf --dry-run
    python ingest.py notes.pdf --model gemma4:e4b --chunk-limit 3
"""

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

# Ensure emoji and non-ASCII characters print safely on all platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import chromadb
import ollama
import requests
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# CONFIGURATION DEFAULTS
# ---------------------------------------------------------------------------

OLLAMA_URL          = "http://localhost:11434/api/generate"
DEFAULT_MODEL       = "gemma4:e4b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION  = "curriculum"
DEFAULT_MAX_CHARS   = 3000
DEFAULT_OVERLAP     = 300   # characters of trailing context carried into the next chunk
TIMEOUT_SECS        = 180   # longer timeout for batch — chunks can be large

# ---------------------------------------------------------------------------
# 1. EXTRACTION
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Read a PDF from disk and return its full text as a single string."""
    path = Path(pdf_path)
    if not path.exists():
        print(f"[ERROR] File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() != ".pdf":
        print(f"[ERROR] Expected a .pdf file, got: {path.suffix}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Extracting text from: {path.name}")
    reader = PdfReader(str(path))
    pages  = [page.extract_text() or "" for page in reader.pages]
    full   = "\n".join(pages).strip()
    print(f"[INFO] Extracted {len(full):,} characters across {len(reader.pages)} page(s).")
    return full


# ---------------------------------------------------------------------------
# 2. CHUNKING
# ---------------------------------------------------------------------------

def semantic_chunk_text(
    text:       str,
    chunk_size: int = DEFAULT_MAX_CHARS,
    overlap:    int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split text into semantically coherent chunks with a trailing overlap window.

    Strategy
    --------
    1. Normalise whitespace and split on double newlines (paragraph / section breaks).
    2. Accumulate paragraphs into a chunk until adding the next one would exceed
       chunk_size.  When the chunk is full:
         a. Save the current chunk.
         b. Seed the next chunk with the trailing `overlap` characters of the saved
            chunk, trimmed to a clean word boundary so the LLM never starts
            mid-word.
    3. Safety valve: if a single paragraph is itself larger than chunk_size it is
       force-sliced to avoid an unbounded chunk.

    The overlap window means a concept that straddles a paragraph boundary is
    always fully present in at least one chunk, preventing the LLM from seeing
    only half of an educational section.
    """
    # Normalise line endings and collapse 3+ blank lines to 2
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            # Still room — append with a paragraph separator
            current_chunk = (current_chunk + "\n\n" + para).lstrip("\n")
        else:
            # Chunk is full — save it
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Build the overlap seed from the tail of the saved chunk,
            # snapped forward to the nearest word boundary
            if overlap and len(current_chunk) > overlap:
                tail = current_chunk[-overlap:]
                # Advance past any partial word at the start of the tail
                tail = tail.split(" ", 1)[-1] if " " in tail else tail
            else:
                tail = current_chunk

            current_chunk = (tail + "\n\n" + para).lstrip("\n")

            # Safety valve: a single paragraph that is itself too large
            if len(current_chunk) > chunk_size:
                chunks.append(current_chunk[:chunk_size].strip())
                current_chunk = current_chunk[chunk_size:]

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    print(
        f"[INFO] Produced {len(chunks)} semantic chunk(s) "
        f"(chunk_size={chunk_size}, overlap={overlap})."
    )
    return chunks


# ---------------------------------------------------------------------------
# 3. LLM CALL
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert pedagogical AI. Your task is to extract educational concepts from the provided text and format them for an interactive learning game.

### EXTRACTION RULES:
1. Do NOT stop after one concept. You MUST extract a distinct concept for EVERY major section or numbered heading in the text.
2. If the text has 4 sections, you must output 4 concepts in the array.

### PERSONA SELECTION RULES:
Analyze the reading level of the provided text to choose the correct persona for the student:
- Primary/Ages 7-10: Persona = "Pip", Age 8. (Curious, uses toy/playground analogies).
- Intermediate/Ages 11-14: Persona = "Alex", Age 13. (Slightly skeptical, uses sports/social/allowance analogies).
- Advanced/Ages 15+: Persona = "Riley", Age 16. (The "Overzealous Detective", invents wild, confident but flawed theories).

### TARGET SCHEMA:
You must return ONLY a raw JSON object matching this exact structure. No markdown formatting.

{
  "extracted_concepts": [
    {
      "concept_name": "Catchy Title",
      "name": "Catchy Title",
      "persona_config": {
        "name": "[Pip, Alex, or Riley]",
        "age": [8, 13, or 16],
        "avatar_emoji": "[👧🏼, 👦🏽, or 🕵️]",
        "voice_tone": "[Brief description of how they speak based on rules above]"
      },
      "story_intro": "The persona presents a confused scenario or flawed theory based on the concept.",
      "ground_truth_logic": "The core fact the user must teach them.",
      "verification_scenario": "A follow-up question where the persona tests their new understanding.",
      "boss_fight_logic": "A logic trap where the persona makes a smart-sounding but incorrect assumption.",
      "home_activity": "A simple real-world physical activity a parent and child can do together to explore this concept."
    }
  ]
}

If the source text does not contain enough content for even one meaningful concept, output: {"skip": true}
"""


def call_ollama(chunk: str, model: str) -> str:
    """Send a chunk to the local Ollama model and return the raw response text."""
    payload = {
        "model":  model,
        "system": SYSTEM_PROMPT,
        "prompt": f"SOURCE TEXT:\n{chunk}\n\nAnalyse complexity, select persona, then generate the concept JSON:",
        "stream": False,
        "format": "json",
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECS)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        print("[WARN] Ollama timed out for this chunk. Skipping.", file=sys.stderr)
        return ""
    except requests.exceptions.ConnectionError:
        print(
            f"[ERROR] Cannot connect to Ollama at {OLLAMA_URL}. "
            "Make sure `ollama serve` is running.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"[WARN] API error for this chunk: {exc}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# 4. JSON PARSING  (with safety net)
# ---------------------------------------------------------------------------

def _validate_concept(data: dict, chunk_index: int, item_index: int) -> dict | None:
    """
    Validate a single concept dict extracted from the LLM array.
    Returns the normalised dict, or None if required fields are missing.
    """
    name = data.get("concept_name") or data.get("name")
    if not name or not data.get("story_intro"):
        print(
            f"[WARN] Chunk {chunk_index}, item {item_index}: "
            "missing required fields (name/story_intro). Skipping item."
        )
        return None

    # Normalise: ensure both 'name' and 'concept_name' are set
    data["name"]         = name
    data["concept_name"] = name

    # Log the detected persona so the user can see the complexity decision
    persona = data.get("persona_config", {})
    if persona:
        print(
            f"[PERSONA] Item {item_index}: complexity={persona.get('complexity_level', '?')} "
            f"-> {persona.get('name', '?')} (age {persona.get('age', '?')}) "
            f"{persona.get('avatar_emoji', '')}"
        )
    else:
        print(f"[WARN] Chunk {chunk_index}, item {item_index}: model did not return a persona_config block.")

    return data


def parse_concepts(raw: str, chunk_index: int) -> list[dict]:
    """
    Parse the LLM's raw output into a list of concept dicts.

    Handles the new multi-concept envelope {"extracted_concepts": [...]} as well
    as the legacy single-object response for robustness.
    Returns an empty list on any unrecoverable parse failure.
    """
    if not raw:
        print(f"[WARN] Chunk {chunk_index}: empty response from model. Skipping.")
        return []

    # Strip accidental markdown fences the model might still add
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"[WARN] Chunk {chunk_index}: JSON parse error — {exc}. Skipping.")
        return []

    # Model signalled there is nothing worth extracting
    if data.get("skip"):
        print(f"[INFO] Chunk {chunk_index}: model signalled insufficient content. Skipping.")
        return []

    # ── New envelope format ──────────────────────────────────────────────────
    if "extracted_concepts" in data:
        raw_list = data["extracted_concepts"]
        if not isinstance(raw_list, list):
            print(f"[WARN] Chunk {chunk_index}: 'extracted_concepts' is not a list. Skipping.")
            return []

        concepts: list[dict] = []
        for idx, item in enumerate(raw_list, start=1):
            if not isinstance(item, dict):
                print(f"[WARN] Chunk {chunk_index}, item {idx}: not a dict. Skipping item.")
                continue
            validated = _validate_concept(item, chunk_index, idx)
            if validated:
                concepts.append(validated)

        print(f"[INFO] Chunk {chunk_index}: extracted {len(concepts)} valid concept(s) from array of {len(raw_list)}.")
        return concepts

    # ── Legacy fallback: model returned a bare single-concept object ─────────
    print(f"[INFO] Chunk {chunk_index}: response is a bare object — treating as single concept (legacy).")
    validated = _validate_concept(data, chunk_index, item_index=1)
    return [validated] if validated else []


# ---------------------------------------------------------------------------
# 5. CHROMADB + EMBEDDING HELPERS
# ---------------------------------------------------------------------------

def get_collection(chroma_path: str, collection_name: str) -> chromadb.Collection:
    """Open (or create) a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=collection_name)
    print(f"[INFO] ChromaDB collection '{collection_name}' ready at '{chroma_path}' "
          f"({collection.count()} existing document(s)).")
    return collection


def _concept_id(concept_name: str) -> str:
    """
    Derive a stable, URL-safe ID from a concept name.
    Identical names always map to the same ID, making upsert naturally idempotent.
    Falls back to a UUID suffix if the name is blank.
    """
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", concept_name.strip()).strip("_")[:60]
    return slug if slug else uuid.uuid4().hex


def _flatten_metadata(concept: dict) -> dict:
    """
    ChromaDB metadata values must be flat scalars (str, int, float, bool).
    Serialize the full concept as a JSON string so it can be reconstructed,
    and surface a few key fields as queryable flat strings.

    All concepts enter as "pending" so the Parent Dashboard can gate them
    before they become part of the active curriculum.
    """
    persona = concept.get("persona_config", {})
    return {
        "concept_json":     json.dumps(concept, ensure_ascii=False),
        "concept_name":     concept.get("name", ""),
        "complexity_level": persona.get("complexity_level", ""),
        "persona_name":     persona.get("name", ""),
        "subject":          concept.get("subject", "General"),
        "sequence_order":   int(concept.get("sequence_order", 999)),
        "status":           "pending",
    }


def embed_and_upsert(
    concept:      dict,
    collection:   chromadb.Collection,
    embed_model:  str,
) -> None:
    """
    Generate a vector embedding for one concept and upsert it into ChromaDB.
    The searchable document combines the concept name + its ground-truth logic.
    """
    concept_name    = concept.get("name", "Unnamed Concept")
    searchable_text = (
        f"{concept_name}: "
        f"{concept.get('ground_truth_logic', concept.get('secret_fact', ''))}"
    )

    try:
        response = ollama.embeddings(model=embed_model, prompt=searchable_text)
        vector   = response["embedding"]
    except Exception as exc:
        print(f"[WARN] Embedding failed for '{concept_name}': {exc}. Skipping.", file=sys.stderr)
        return

    doc_id = _concept_id(concept_name)

    collection.upsert(
        ids        = [doc_id],
        embeddings = [vector],
        documents  = [searchable_text],
        metadatas  = [_flatten_metadata(concept)],
    )
    print(f"[OK] Embedded and upserted '{concept_name}' (id={doc_id}).")


# ---------------------------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run(
    pdf_path:        str,
    model:           str,
    embed_model:     str,
    chroma_path:     str,
    collection_name: str,
    max_chars:       int,
    overlap:         int,
    chunk_limit:     int | None,
    dry_run:         bool,
    subject:         str = "General",
) -> None:
    print("=" * 60)
    print("GemmaGenius — PDF Ingestion Pipeline (ChromaDB)")
    if dry_run:
        print("  *** DRY RUN MODE — nothing will be embedded or written ***")
    print("=" * 60)

    # Step 1 — Extract text from PDF
    full_text = extract_text(pdf_path)
    if len(full_text) < 200:
        print("[ERROR] Extracted text is too short to be useful. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Step 2 — Semantic chunking with overlap
    chunks = semantic_chunk_text(full_text, chunk_size=max_chars, overlap=overlap)
    if chunk_limit:
        chunks = chunks[:chunk_limit]
        print(f"[INFO] Processing first {len(chunks)} chunk(s) (--chunk-limit applied).")

    # Step 3 — Connect to ChromaDB (skip in dry-run to avoid creating an empty store)
    collection = None
    if not dry_run:
        collection = get_collection(chroma_path, collection_name)

    embedded         = 0
    skipped          = 0
    current_sequence = 1   # global sequence across all chunks — tracks linear order

    for i, chunk in enumerate(chunks, start=1):
        print(f"\nProcessing logical chunk {i} of {len(chunks)}…  (sending to {model})")

        # Step 4a — Generate structured concept JSON via Gemma
        raw      = call_ollama(chunk, model)
        concepts = parse_concepts(raw, chunk_index=i)

        if not concepts:
            skipped += 1
            continue

        for concept in concepts:
            # Stamp subject and linear sequence order before embedding / dry-run output
            concept["subject"]        = subject
            concept["sequence_order"] = current_sequence

            if dry_run:
                print(f"\n--- DRY RUN OUTPUT (chunk {i}, sequence {current_sequence}) ---")
                print(json.dumps(concept, indent=2, ensure_ascii=False))
                print("----------------------------------")
                embedded += 1
                current_sequence += 1
                continue

            # Step 4b — Embed with nomic-embed-text and upsert to ChromaDB
            embed_and_upsert(concept, collection, embed_model)
            embedded += 1

            current_sequence += 1  # increment after each valid concept

        # Brief pause between chunks to avoid hammering the local model
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print(f"Done.  Embedded: {embedded}   Skipped (empty/error): {skipped}   "
          f"Total chunks: {len(chunks)}")
    if not dry_run and embedded > 0:
        print(f"Concepts are stored in ChromaDB at '{chroma_path}' "
              f"(collection: '{collection_name}').")
    if dry_run:
        print("Dry run complete — no data was written to ChromaDB.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-ingest a PDF into GemmaGenius ChromaDB via local Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pdf_path",
                   help="Path to the PDF file to ingest.")
    p.add_argument("--dry-run",       action="store_true",
                   help="Print generated JSON to terminal; do NOT embed or write to ChromaDB.")
    p.add_argument("--model",         default=DEFAULT_MODEL,
                   help=f"Ollama generation model (default: {DEFAULT_MODEL}).")
    p.add_argument("--embed-model",   default=DEFAULT_EMBED_MODEL,
                   help=f"Ollama embedding model (default: {DEFAULT_EMBED_MODEL}).")
    p.add_argument("--chroma-path",   default=DEFAULT_CHROMA_PATH,
                   help=f"Directory for the ChromaDB store (default: {DEFAULT_CHROMA_PATH}).")
    p.add_argument("--collection",    default=DEFAULT_COLLECTION,
                   help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION}).")
    p.add_argument("--max-chars",     type=int, default=DEFAULT_MAX_CHARS,
                   help=f"Max characters per chunk (default: {DEFAULT_MAX_CHARS}).")
    p.add_argument("--overlap",       type=int, default=DEFAULT_OVERLAP,
                   help=f"Overlap characters carried into each new chunk (default: {DEFAULT_OVERLAP}).")
    p.add_argument("--chunk-limit",   type=int, default=None,
                   help="Process at most N chunks (default: all).")
    p.add_argument("--subject",       type=str, default="General",
                   help="Subject category for the curriculum (e.g., Math, Science, Finance). "
                        "Default: General.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        pdf_path        = args.pdf_path,
        model           = args.model,
        embed_model     = args.embed_model,
        chroma_path     = args.chroma_path,
        collection_name = args.collection,
        max_chars       = args.max_chars,
        overlap         = args.overlap,
        chunk_limit     = args.chunk_limit,
        dry_run         = args.dry_run,
        subject         = args.subject,
    )
