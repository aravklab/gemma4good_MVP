"""
ingest.py - Batch PDF to knowledge.json pipeline for GemmaGenius
=================================================================
Reads a PDF, chunks it into paragraph blocks, sends each chunk to a
local Ollama model, and appends the generated concept(s) to knowledge.json.

Usage:
    python ingest.py <path/to/file.pdf> [options]

Options:
    --dry-run         Print generated JSON to terminal; do NOT write to file.
    --model NAME      Ollama model to use (default: gemma4:e4b).
    --knowledge PATH  Path to knowledge.json (default: knowledge.json).
    --max-chars N     Max characters per chunk (default: 3000).
    --chunk-limit N   Process at most N chunks (default: all).

Examples:
    python ingest.py textbook_chapter.pdf --dry-run
    python ingest.py notes.pdf --model llama3 --chunk-limit 3
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Ensure emoji and non-ASCII characters print safely on all platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# CONFIGURATION DEFAULTS
# ---------------------------------------------------------------------------

OLLAMA_URL      = "http://localhost:11434/api/generate"
DEFAULT_MODEL   = "gemma4:e4b"
DEFAULT_KG_PATH = "knowledge.json"
DEFAULT_MAX_CHARS = 3000
TIMEOUT_SECS    = 180   # longer timeout for batch — chunks can be large

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

def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """
    Split text into logical blocks no larger than max_chars.

    Strategy:
      1. Split on double newlines (paragraph breaks).
      2. If a paragraph itself exceeds max_chars, split further on
         single newlines, then on sentence boundaries.
      3. Merge small consecutive paragraphs up to the max_chars limit.
    """
    # Normalise whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    raw_paragraphs: list[str] = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Sub-split any paragraph that exceeds max_chars
    split_paragraphs: list[str] = []
    for para in raw_paragraphs:
        if len(para) <= max_chars:
            split_paragraphs.append(para)
        else:
            # Try single-newline splits first
            lines = [l.strip() for l in para.split("\n") if l.strip()]
            current = ""
            for line in lines:
                if len(current) + len(line) + 1 <= max_chars:
                    current = f"{current}\n{line}".strip()
                else:
                    if current:
                        split_paragraphs.append(current)
                    # If a single line is still too long, cut by sentence
                    if len(line) > max_chars:
                        sentences = re.split(r"(?<=[.!?])\s+", line)
                        buf = ""
                        for sent in sentences:
                            if len(buf) + len(sent) + 1 <= max_chars:
                                buf = f"{buf} {sent}".strip()
                            else:
                                if buf:
                                    split_paragraphs.append(buf)
                                buf = sent[:max_chars]
                        if buf:
                            split_paragraphs.append(buf)
                    else:
                        current = line
            if current:
                split_paragraphs.append(current)

    # Merge consecutive small paragraphs
    chunks: list[str] = []
    buffer = ""
    for para in split_paragraphs:
        if len(buffer) + len(para) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{para}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para
    if buffer:
        chunks.append(buffer)

    print(f"[INFO] Produced {len(chunks)} chunk(s) from the extracted text.")
    return chunks


# ---------------------------------------------------------------------------
# 3. LLM CALL
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert curriculum designer for GemmaGenius, a Feynman Technique learning app.

TASK: Read the provided text and identify the CORE distinct educational concepts.
Extract between 1 and 3 concepts maximum (ignore minor trivia or repeated ideas).

STEP 1 — COMPLEXITY DETECTION:
Before writing any content, analyse the source text and classify it as exactly one of:
  - "Primary"      : simple facts, basic vocabulary, early-school level
  - "Intermediate" : multi-step reasoning, middle-school vocabulary, some domain terms
  - "Advanced"     : technical mechanisms, high-school/university vocabulary, abstract logic

STEP 2 — PERSONA SELECTION:
Based on the complexity level, select ONE persona that applies to ALL concepts in this chunk:

  Primary   -> { "name": "Pip",  "age": 8,  "complexity_level": "Primary",
                 "avatar_emoji": "🧒",
                 "voice_tone": "Curious 8-year-old who uses toy and playground analogies" }

  Intermediate -> { "name": "Alex", "age": 13, "complexity_level": "Intermediate",
                    "avatar_emoji": "🧑‍🏫",
                    "voice_tone": "Slightly skeptical 13-year-old who uses sports and social analogies" }

  Advanced  -> { "name": "Riley", "age": 16, "complexity_level": "Advanced",
                 "avatar_emoji": "🕵️",
                 "voice_tone": "Overzealous detective who invents wild, confidently incorrect theories and demands the user confirm or debunk them" }

STEP 3 — WRITE ALL STORY FIELDS for EACH concept in the selected persona's voice.
The persona must be consistent across story_intro, story_bridge, variants, and verification_scenario
for every concept in the array.

TARGET SCHEMA:
{
  "extracted_concepts": [
    {
      "persona_config": {
        "name": "<Pip | Alex | Riley>",
        "age": <8 | 13 | 16>,
        "complexity_level": "<Primary | Intermediate | Advanced>",
        "avatar_emoji": "<emoji>",
        "voice_tone": "<tone description>"
      },
      "concept_name": "Catchy Title",
      "name": "Catchy Title",
      "story_intro": "<persona> is in a real-world scenario confused by this topic.",
      "story_bridge": "<persona>'s brief celebration when the student explains it correctly.",
      "secret_fact": "The actual scientific or mathematical explanation in one sentence.",
      "goal": "What the student must understand to win.",
      "evaluator_ground_truth": "Phase 1 ground truth: what the student must explain.",
      "variants": [
        {
          "story_intro": "Alternate scenario A in <persona>'s voice (same concept, different context).",
          "verification_scenario": "<persona> applies the logic but makes a specific logical error. Ends with 'right?'",
          "verification_ground_truth": "Why the variant scenario is wrong."
        },
        {
          "story_intro": "Alternate scenario B in <persona>'s voice.",
          "verification_scenario": "Another logical trap <persona> falls into.",
          "verification_ground_truth": "Why this variant scenario is wrong."
        }
      ],
      "verification_scenario": "<persona> applies the concept but makes a specific error. Ends with 'right?'",
      "verification_ground_truth": "The specific error the student must catch and correct.",
      "ground_truth_logic": "The actual scientific or mathematical explanation.",
      "boss_fight_logic": "The specific misconception the student must debunk in Phase 2.",
      "home_activity": "A simple real-world physical activity a parent and child can do together to explore this concept."
    }
  ]
}

RULES:
- Return an array of 1 to 3 concept objects inside "extracted_concepts". Never return more than 3.
- ALL story text must match the chosen persona's age and voice_tone. Do not mix personas.
- The verification_scenario must be a logical trap containing one clear correctable error.
- RILEY-SPECIFIC RULE: When the persona is Riley, story_intro and verification_scenario must NOT be
  simple questions. Riley must confidently present a wild, over-the-top, elaborately wrong theory
  derived from the text — like a detective who has "cracked the case" but got it completely wrong.
  Riley is energetic and sassy. End every Riley verification_scenario with "I've cracked the code,
  haven't I?" instead of the standard "right?".
- CRITICAL: Return ONLY the raw JSON object containing the 'extracted_concepts' array.
  Do NOT wrap it in markdown formatting or add any conversational text before or after.
- If the source text does not contain enough content for even one meaningful concept,
  output: {"skip": true}
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
# 5. KNOWLEDGE.JSON HELPERS
# ---------------------------------------------------------------------------

def load_knowledge(path: str) -> dict:
    """Load knowledge.json; return an empty structure if the file is missing."""
    kp = Path(path)
    if not kp.exists():
        print(f"[INFO] {path} not found — will create a new file.")
        return {"concepts": {}}
    try:
        with kp.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] {path} is malformed: {exc}", file=sys.stderr)
        sys.exit(1)


def save_knowledge(knowledge: dict, path: str) -> None:
    """Write the knowledge dict back to disk."""
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(knowledge, fh, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {path}")


def make_concept_key(name: str, existing_keys: list[str]) -> str:
    """Generate a unique CamelCase-ish key for the concepts dict."""
    sanitized = "_".join(name.split())[:30]
    n         = len(existing_keys) + 1
    candidate = f"C{n}_{sanitized}"
    while candidate in existing_keys:
        n        += 1
        candidate = f"C{n}_{sanitized}"
    return candidate


def is_duplicate(concept_name: str, knowledge: dict) -> bool:
    """Return True if concept_name already exists in knowledge.json."""
    existing_names = {
        v.get("name", "").lower()
        for v in knowledge.get("concepts", {}).values()
    }
    return concept_name.lower() in existing_names


# ---------------------------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run(
    pdf_path:    str,
    model:       str,
    kg_path:     str,
    max_chars:   int,
    chunk_limit: int | None,
    dry_run:     bool,
) -> None:
    print("=" * 60)
    print("GemmaGenius — PDF Ingestion Pipeline")
    if dry_run:
        print("  *** DRY RUN MODE — nothing will be written to disk ***")
    print("=" * 60)

    # Step 1 — Extract
    full_text = extract_text(pdf_path)
    if len(full_text) < 200:
        print("[ERROR] Extracted text is too short to be useful. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Step 2 — Chunk
    chunks = chunk_text(full_text, max_chars=max_chars)
    if chunk_limit:
        chunks = chunks[:chunk_limit]
        print(f"[INFO] Processing first {len(chunks)} chunk(s) (--chunk-limit applied).")

    # Step 3 — Load existing knowledge (needed for duplicate check even in dry-run)
    knowledge = load_knowledge(kg_path)

    added   = 0
    skipped = 0

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n[{i}/{len(chunks)}] Sending chunk to {model}…")

        raw      = call_ollama(chunk, model)
        concepts = parse_concepts(raw, chunk_index=i)

        if not concepts:
            skipped += 1
            continue

        for concept in concepts:
            concept_name = concept["name"]

            # Duplicate guard (re-checked after every addition so keys stay fresh)
            if is_duplicate(concept_name, knowledge):
                print(f"[SKIP] '{concept_name}' already exists in {kg_path}. Skipping.")
                skipped += 1
                continue

            if dry_run:
                print(f"\n--- DRY RUN OUTPUT (chunk {i}) ---")
                print(json.dumps(concept, indent=2, ensure_ascii=False))
                print("----------------------------------")
                added += 1
                continue

            # Append to knowledge
            key                        = make_concept_key(concept_name, list(knowledge["concepts"].keys()))
            knowledge["concepts"][key] = concept
            print(f"[OK] Added concept '{concept_name}' as key '{key}'.")
            added += 1

        # Brief pause between chunks to avoid hammering the local model
        time.sleep(0.5)

    # Step 5 — Save (unless dry-run)
    if not dry_run and added > 0:
        save_knowledge(knowledge, kg_path)

    print("\n" + "=" * 60)
    print(f"Done.  Added: {added}   Skipped: {skipped}   Total chunks: {len(chunks)}")
    if dry_run:
        print("Dry run complete — no files were modified.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-ingest a PDF into GemmaGenius knowledge.json via local Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pdf_path",                  help="Path to the PDF file to ingest.")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print generated JSON to terminal; do NOT write to knowledge.json.")
    p.add_argument("--model",     default=DEFAULT_MODEL,
                   help=f"Ollama model to use (default: {DEFAULT_MODEL}).")
    p.add_argument("--knowledge", default=DEFAULT_KG_PATH,
                   help=f"Path to knowledge.json (default: {DEFAULT_KG_PATH}).")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                   help=f"Max characters per chunk (default: {DEFAULT_MAX_CHARS}).")
    p.add_argument("--chunk-limit", type=int, default=None,
                   help="Process at most N chunks (default: all).")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        pdf_path    = args.pdf_path,
        model       = args.model,
        kg_path     = args.knowledge,
        max_chars   = args.max_chars,
        chunk_limit = args.chunk_limit,
        dry_run     = args.dry_run,
    )
