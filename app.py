"""
app.py — GemmaGenius Streamlit Web UI
======================================
PURPOSE
-------
The main application. Runs the full GemmaGenius learning experience:
two views (Kid Mode and Parent Dashboard) sharing one Streamlit process.

Run with:
    streamlit run app.py   (from the Gemma4good/ directory)

ARCHITECTURE OVERVIEW
---------------------
Two AI units, one local Ollama model, zero cloud calls:

    ┌─────────────────────────────────────────────────────────┐
    │  KID MODE (Play)            PARENT DASHBOARD            │
    │  ─────────────────          ─────────────────────────── │
    │  Subject folder nav         Review Queue (pending)      │
    │  Look-ahead locking         Active Curriculum manager   │
    │  JIT level generation       PDF ingestion + sequencing  │
    │  Two-phase Feynman loop     Analytics + Trophy Room     │
    │  Trophy Room                Sequence repair tool        │
    └─────────────────────────────────────────────────────────┘

JIT LEVEL GENERATION (Just-In-Time)
-------------------------------------
When a student clicks an unlocked concept, app.py checks whether the concept
already has a story (legacy knowledge.json) or is a bare skeleton (ChromaDB).

  Skeleton concept (from ingest.py):
    concept_obj has concept_name + ground_truth_logic only
    → generate_level_jit() calls Ollama to generate:
        persona_config, story_intro, verification_scenario,
        boss_fight_logic, home_activity
    → every click produces a FRESH story (dynamic re-playability)
    → merged onto the skeleton dict → start_session()

  Legacy concept (from knowledge.json):
    concept_obj already has story_intro
    → used as-is with optional Scenario Polymorphism (random variant)
    → start_session()

TWO-PHASE GAME LOOP
-------------------
Phase 1 — Elicitation
  Persona presents story_intro and asks the student to explain it.
  Evaluator grades each response: mastery / partial_hit / miss /
  question / off_topic / give_up.
  On mastery → seamless transition to Phase 2.

Phase 2 — Boss Fight (Verification)
  Persona presents verification_scenario with a deliberate mistake.
  Student must identify and correct the error.
  On mastery → TRUE MASTERY ACHIEVED → achievement saved to disk.

DUAL-AGENT DESIGN
-----------------
Both agents share the same Ollama model but have completely separate
system prompts and roles:

  The Evaluator  — silent judge, outputs only {"classification": "…"}
                   never reveals its verdict to the student
  The Persona    — actor (Pip / Alex / Riley), never gives answers,
                   adapts its voice from persona_config in concept_data

SUBJECT-GROUPED SIDEBAR WITH LOOK-AHEAD LOCKING
-------------------------------------------------
Concepts are grouped into subject folders (📁 Biology, 📁 Classic …).
Within each folder, concepts are sorted by sequence_order (float, fractional).
Locking uses a two-pass look-ahead buffer (LOOK_AHEAD = 3):
  Pass 1: find the highest mastered sequence in the folder
  Pass 2: unlock anything with seq <= max_mastered + 3
  Concepts stored before sequencing (seq=999) are always unlocked.

DATA FLOW
---------
  ChromaDB  →  approved concepts  →  merged with knowledge.json legacy
            →  sidebar folder UI
            →  JIT generation on click
            →  st.session_state.concept_data (ephemeral per session)

  student_profile.json  →  achievements (mastery records, keyed by slug ID)
                        →  Trophy Room display
                        →  locking gate check (cid in achievements)

KEY SESSION STATE KEYS
-----------------------
  messages              Full chat history (list of role/content dicts)
  current_phase         1 = Elicitation, 2 = Boss Fight
  concept_data          Active concept dict (populated by JIT or legacy path)
  current_concept_id    ChromaDB slug ID of the active concept
  frustration_counter   Phase 1 miss count (used for Bridging Constraint)
  awaiting_retry        True after give_up — shows Yes/No buttons
  profile               Student achievements (loaded from student_profile.json)
  game_started          True once Start/Reset has been triggered
"""

import io
import json
import os
import random
import re
import requests
import tempfile
from datetime import datetime, timezone
import chromadb
import ollama
import streamlit as st
from pypdf import PdfReader

try:
    from audio_recorder_streamlit import audio_recorder as _audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel as _WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

OLLAMA_URL       = "http://localhost:11434/api/generate"
MODEL_NAME       = "gemma4:e4b"
EMBED_MODEL      = "nomic-embed-text"
TIMEOUT_SECS     = 120
PROFILE_FILE     = "student_profile.json"
KNOWLEDGE_FILE   = "knowledge.json"
CHROMA_PATH       = "./chroma_db"
CHROMA_COLLECTION = "curriculum"
DLQ_FILE          = "rejected_telemetry.jsonl"  # Dead Letter Queue — one JSON record per line
PDF_MIN_CHARS     = 500            # fail-fast guardrail threshold


# ---------------------------------------------------------------------------
# A. PERSISTENCE LAYER
# ---------------------------------------------------------------------------

def load_profile() -> dict:
    """Load student_profile.json; fall back to a fresh default if missing or corrupt."""
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {"student_name": "Explorer", "achievements": {}}


def save_profile(profile_data: dict) -> None:
    """Write the current profile dict to student_profile.json."""
    try:
        with open(PROFILE_FILE, "w", encoding="utf-8") as fh:
            json.dump(profile_data, fh, indent=2, ensure_ascii=False)
    except OSError:
        st.warning("⚠️ Could not save profile — check file permissions.", icon="⚠️")


def load_knowledge() -> dict:
    """Load knowledge.json from disk."""
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"concepts": {}}


def save_knowledge(knowledge: dict) -> None:
    """Persist knowledge dict to knowledge.json."""
    try:
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as fh:
            json.dump(knowledge, fh, indent=2, ensure_ascii=False)
    except OSError:
        st.error("❌ Could not save knowledge.json — check file permissions.")


# ---------------------------------------------------------------------------
# CHROMADB HELPERS
# ---------------------------------------------------------------------------

@st.cache_resource
def get_chroma_collection() -> chromadb.Collection:
    """
    Open (or create) the persistent ChromaDB curriculum collection.
    Cached by Streamlit so the client is reused across reruns.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=CHROMA_COLLECTION)


def _chroma_concept_id(concept_name: str) -> str:
    """Derive a stable, URL-safe ChromaDB document ID from a concept name."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", concept_name.strip()).strip("_")[:60]
    return slug if slug else "unnamed_concept"


def _embed_and_upsert_concept(concept: dict, status: str = "pending") -> bool:
    """
    Generate a vector for the concept and upsert it into ChromaDB.
    Returns True on success, False on failure (shows a Streamlit warning).
    """
    concept_name    = concept.get("name", "Untitled Concept")
    searchable_text = (
        f"{concept_name}: "
        f"{concept.get('ground_truth_logic', concept.get('secret_fact', ''))}"
    )
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=searchable_text)
        vector   = response["embedding"]
    except Exception as exc:
        st.warning(f"⚠️ Embedding failed for '{concept_name}': {exc}")
        return False

    persona   = concept.get("persona_config", {})
    metadata  = {
        "concept_json":     json.dumps(concept, ensure_ascii=False),
        "concept_name":     concept_name,
        "complexity_level": persona.get("complexity_level", ""),
        "persona_name":     persona.get("name", ""),
        "subject":          concept.get("subject", "General"),
        "sequence_order":   float(concept.get("sequence_order", 999)),
        "status":           status,
    }
    collection = get_chroma_collection()
    collection.upsert(
        ids        = [_chroma_concept_id(concept_name)],
        embeddings = [vector],
        documents  = [searchable_text],
        metadatas  = [metadata],
    )
    return True


def write_dlq(item_id: str, concept: dict, rejection_reason: str) -> None:
    """
    Append one record to the Dead Letter Queue (dlq.jsonl).
    Each line is a self-contained JSON object so the file can be streamed
    or analysed with standard tools (jq, pandas, etc.).
    """
    record = {
        "chroma_id":        item_id,
        "rejected_at":      datetime.now(timezone.utc).isoformat(),
        "rejection_reason": rejection_reason,
        "concept":          concept,
    }
    try:
        with open(DLQ_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        st.warning(f"⚠️ Could not write to DLQ ({DLQ_FILE}): {exc}")


# ---------------------------------------------------------------------------
# PDF INGESTION HELPERS
# ---------------------------------------------------------------------------

def extract_pdf_text(uploaded_file) -> str:
    """Return all text from an uploaded PDF file object."""
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def compile_concept_generation_prompt(extracted_text: str) -> tuple[str, str]:
    """
    Build the (system, user) prompt that turns raw PDF text into an array of
    GemmaGenius concept objects — one per major section of the source material.
    """
    system_prompt = """\
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
    user_prompt = (
        f"SOURCE MATERIAL:\n{extracted_text[:8000]}\n\n"
        "Extract all key concepts and return the JSON:"
    )
    return system_prompt, user_prompt


def generate_level_jit(
    concept_name:      str,
    ground_truth_logic: str,
    skeleton:          dict | None = None,
) -> dict | None:
    """
    Just-In-Time level generator.
    Takes the skeleton from ChromaDB and calls Ollama to generate the full
    playable game level (persona, story, boss fight) on demand.
    Returns a merged concept dict ready for start_session(), or None on failure.
    """
    loading_messages = [
        "Pip is putting on her thinking cap... 🧢",
        "Loading the boss fight logic... 👾",
        "Sprinkling some puzzle dust... ✨",
        "Waking up the local AI... 🤖",
        "Brewing the perfect puzzle... ☕",
    ]

    jit_system = """\
You are a game level designer for GemmaGenius, an educational app for children.

### PERSONA SELECTION RULES:
Choose a persona based on the complexity of the ground truth:
- Simple/concrete facts → Persona = "Pip", Age 8, avatar_emoji "👧🏼", curious 8-year-old who uses toy and playground analogies.
- Moderately complex → Persona = "Alex", Age 13, avatar_emoji "👦🏽", slightly skeptical, uses sports/social/allowance analogies.
- Advanced/abstract → Persona = "Riley", Age 16, avatar_emoji "🕵️", overzealous detective who invents wild confident-but-wrong theories.

### OUTPUT RULES:
Return ONLY a valid JSON object. No markdown. No extra keys.

{
  "persona_config": {
    "name": "[Pip, Alex, or Riley]",
    "age": [8, 13, or 16],
    "avatar_emoji": "[👧🏼, 👦🏽, or 🕵️]",
    "voice_tone": "Brief description matching the persona rules above"
  },
  "story_intro": "The persona presents a confused scenario about the concept, in character, ending with a question.",
  "verification_scenario": "A follow-up where the persona tests their understanding by applying the concept — but makes a subtle logical mistake. Ends with 'right?'",
  "boss_fight_logic": "The exact wrong assumption the persona makes in verification_scenario that the student must correct.",
  "home_activity": "A simple real-world parent-child activity to reinforce this concept."
}
"""

    jit_user = (
        f"CONCEPT: {concept_name}\n"
        f"CORE FACT: {ground_truth_logic}\n\n"
        "Generate the game level JSON:"
    )

    with st.spinner(random.choice(loading_messages)):
        raw = call_ollama(jit_system, jit_user, json_mode=True)

    # Strip markdown fences just in case
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()

    try:
        generated = json.loads(clean)
    except json.JSONDecodeError:
        return None

    # Merge generated layer onto the skeleton so all metadata is preserved
    base = (skeleton or {}).copy()
    base.update(generated)
    base.setdefault("name", concept_name)
    base.setdefault("concept_name", concept_name)
    base.setdefault("ground_truth_logic", ground_truth_logic)
    return base


def make_concept_key(name: str, existing_keys: list) -> str:
    """Produce a unique snake_case key for a new concept entry."""
    sanitized = "_".join(name.split())[:30]
    n         = len(existing_keys) + 1
    candidate = f"C{n}_{sanitized}"
    while candidate in existing_keys:
        n        += 1
        candidate = f"C{n}_{sanitized}"
    return candidate


# ---------------------------------------------------------------------------
# B. API UTILITY
# ---------------------------------------------------------------------------

def call_ollama(
    system_prompt: str,
    user_prompt:   str,
    json_mode:     bool = False,
) -> str:
    """
    POST to the local Ollama /api/generate endpoint (non-streaming).

    Streaming is omitted because Streamlit manages its own render loop;
    the spinner replaces the CLI's real-time token output.
    """
    payload = {
        "model":  MODEL_NAME,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"

    # Capture full prompt for the developer console before every call
    st.session_state.last_prompt = (
        f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    )

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECS)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        st.session_state.last_response = raw   # capture before any parsing
        return raw

    except requests.exceptions.Timeout:
        err = "[System] Ollama timed out. Is `ollama serve` running?"
        st.session_state.last_response = err
        return err
    except requests.exceptions.ConnectionError:
        err = (
            f"[System] Could not connect to Ollama at {OLLAMA_URL}. "
            "Start it with: `ollama serve`"
        )
        st.session_state.last_response = err
        return err
    except requests.exceptions.RequestException as exc:
        err = f"[System] Unexpected API error: {exc}"
        st.session_state.last_response = err
        return err


# ---------------------------------------------------------------------------
# C. PROMPT COMPILERS
# ---------------------------------------------------------------------------

def compile_evaluator_prompt(
    concept_data:       dict,
    latest_user_input:  str,
    pips_last_question: str,
    phase:              int = 1,
) -> tuple[str, str]:
    """
    Build the (system, user) prompt pair for The Evaluator.
    phase=1 → grade against evaluator_ground_truth  (Elicitation)
    phase=2 → grade against verification_ground_truth (Boss Fight)
    """
    # Support both schema generations:
    #   Legacy (knowledge.json): evaluator_ground_truth / verification_ground_truth
    #   ChromaDB (ingest.py):    ground_truth_logic      / boss_fight_logic
    ground_truth = (
        concept_data.get("verification_ground_truth")
        or concept_data.get("boss_fight_logic", "")
        if phase == 2
        else concept_data.get("evaluator_ground_truth")
        or concept_data.get("ground_truth_logic", "")
    )

    system_prompt = f"""You are a strict, impartial pedagogical grader.
Your ONLY job is to classify the student's latest response.

GROUND TRUTH (what the student must ultimately convey):
{ground_truth}

OUTPUT RULES — you must output ONLY valid JSON, nothing else:
{{"classification": "<value>"}}

Choose exactly ONE value from this list:

  "mastery"     — Student clearly explained the core mechanism required by the ground truth. (Childlike language is fine).
  "partial_hit" — Student mentions relevant ideas or vocabulary, but the explanation is incomplete, vague, or missing the core logical mechanism defined in the ground truth.
  "miss"        — Student is wrong, confused, guessing blindly, or merely parroting the vocabulary word without explaining how it actually works.
  "question"    — Student is asking a clarifying question, OR asking for a hint, clue, or nudge (e.g. "can you give me a hint?", "a little hint please", "give me a clue"). Hint requests are ALWAYS "question", never "give_up".
  "off_topic"   — Student is joking, typing nonsense, or talking about something completely unrelated to the current task.
  "give_up"     — Student explicitly and unambiguously surrenders: says "I give up", "I don't know, just tell me", "you explain it", "I quit", or uses language that clearly means they want to stop trying entirely.

Be strict but fair. Do not guess intent — classify what was actually said.
CRITICAL RULES:
- Asking for a hint is ALWAYS "question". Never classify hint requests as "give_up".
- A wrong answer is always "miss", not "give_up".
- Only use "give_up" when the student clearly and explicitly quits with no attempt remaining.
"""

    user_prompt = (
        f"PIP'S LAST QUESTION:\n{pips_last_question}\n\n"
        f"STUDENT'S LATEST RESPONSE:\n{latest_user_input}\n\n"
        "Your JSON classification:"
    )

    return system_prompt, user_prompt


def compile_pip_prompt(concept_data: dict, state_directive: str) -> str:
    """Build the persona's system prompt, injecting voice from persona_config each turn."""
    persona = concept_data.get("persona_config", {
        "name":       "Pip",
        "age":        8,
        "voice_tone": "curious 8-year-old, uses simple toy and playground analogies",
    })
    name       = persona.get("name", "Pip")
    age        = persona.get("age", 8)
    voice_tone = persona.get("voice_tone", "curious 8-year-old, uses simple analogies")

    return f"""You are {name}, a {age}-year-old trying to solve a puzzle.
Your voice and personality: {voice_tone}.
The context of your puzzle is based on the initial story you were told.
You are talking to a friend who is helping you figure it out.

INFORMATION FIREWALL — THIS IS YOUR MOST IMPORTANT RULE:
CRITICAL: Unless your specific State Directive explicitly tells you to give the answer,
you must NEVER give away the exact answer or vocabulary word. Wait for the user to explain it to you.

SPEAKING RULES:
- Maximum 3 short, simple sentences per reply.
- Fully adopt the voice and personality described above — no textbook language.
- Stay in character as the confused puzzle-solver at all times.
- End each reply with exactly ONE question to keep the conversation going.

--- CURRENT INSTRUCTION (follow this for this turn ONLY) ---
{state_directive}
"""


# ---------------------------------------------------------------------------
# D. ROUTING LOGIC
# ---------------------------------------------------------------------------

def build_state_directive(
    classification:        str,
    concept_data:          dict,
    current_phase:         int,
    frustration_counter:   int,
    clarification_counter: int,
    latest_user_input:     str = "",
) -> tuple[str, int, int]:
    """
    Map the Evaluator's classification to Pip's state_directive.

    Returns
    -------
    (state_directive, new_frustration_counter, new_clarification_counter)
    """
    new_frustration   = frustration_counter
    new_clarification = clarification_counter

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1 — ELICITATION
    # ════════════════════════════════════════════════════════════════════
    if current_phase == 1:

        if classification == "partial_hit":
            directive = (
                "The user has part of the right idea, but their explanation is incomplete. "
                "Enthusiastically validate what they got right, but ask a guiding question "
                "to help them figure out the missing piece. Do not give away the exact answer."
            )

        elif classification == "miss":
            new_frustration += 1
            current_intro = concept_data.get("story_intro", "my puzzle")
            directive = f"""\
You are playing your persona. You are currently confused about this specific problem:
"{current_intro}"

The user just tried to help by saying:
"{latest_user_input}"

This answer is either incorrect or doesn't make sense for your problem.

YOUR TASK:
Respond to the user in character. You must do these three things in order:
1. Acknowledge what the user just said.
2. Explain why that doesn't fix your specific problem (Bridge it back to your original confusion).
3. Ask them to try explaining it again.

CRITICAL RULES:
- Do NOT invent new topics, materials, or scenarios.
- Stay fiercely anchored to your original problem.
- Keep your response under 3 sentences.\
"""

        elif classification == "off_topic":
            puzzle = concept_data.get("story_intro", "my puzzle")
            directive = (
                f"The user is distracted. Acknowledge what they said like a kid would, "
                f"then immediately remind them of your exact problem: '{puzzle}'. "
                f"Ask them to help you figure it out."
            )

        elif classification == "question":
            if clarification_counter < 2:
                new_clarification += 1
                puzzle = concept_data.get("story_intro", "my puzzle")
                directive = (
                    f"The user asked a question or is confused. Answer them briefly "
                    f"in character. Then, you MUST remind them of your exact problem: "
                    f"'{puzzle}'. Do not give away the answer."
                )
            else:
                directive = (
                    "The user keeps asking questions and you are both going in circles. "
                    "Tell them you are too confused too, and suggest that maybe you both "
                    "need to look at an actual book together to figure it out."
                )

        elif classification == "give_up":
            analogy = (
                concept_data.get("secret_fact")
                or concept_data.get("ground_truth_logic", "the core concept")
            )
            directive = (
                f"OVERRIDE FIREWALL: The user is frustrated and gave up. Take a deep "
                f"breath and gently explain the answer to them in a full, helpful paragraph. "
                f"Use this specific analogy to make it make sense: '{analogy}'. After you "
                f"finish explaining, you MUST end your response by asking a strict Yes or No "
                f"question: 'Now that we know the secret, do you want to try explaining it "
                f"to me so we can finish?'"
            )

        else:
            directive = (
                "The user said something unexpected. Gently bring the conversation "
                "back to your original puzzle."
            )

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2 — VERIFICATION BOSS FIGHT
    # ════════════════════════════════════════════════════════════════════
    else:

        if classification == "partial_hit":
            directive = (
                "The user's debugging logic is partially correct but incomplete. "
                "Validate what they got right, and ask a follow-up question to help "
                "them fully debunk your flawed scenario."
            )

        elif classification == "miss":
            directive = (
                "The user didn't catch your mistake. Act genuinely confused about "
                "your own scenario and ask a specific follow-up question that nudges "
                "them toward finding the flaw in your logic."
            )

        elif classification == "off_topic":
            puzzle = concept_data.get("verification_scenario", "my scenario")
            directive = (
                f"The user is distracted. Acknowledge what they said like a kid would, "
                f"then immediately remind them of your exact scenario: '{puzzle}'. "
                f"Ask them if your idea is right or wrong."
            )

        elif classification == "question":
            if clarification_counter < 2:
                new_clarification += 1
                puzzle = concept_data.get("verification_scenario", "my scenario")
                directive = (
                    f"The user asked a question or is confused. Answer them briefly "
                    f"in character. Then, you MUST remind them of your exact scenario: "
                    f"'{puzzle}'. Do not give away the answer."
                )
            else:
                directive = (
                    "You and the user are both confused. Suggest you draw it out "
                    "on paper together to see what the scenario would actually look like."
                )

        elif classification == "give_up":
            ans = (
                concept_data.get("verification_ground_truth")
                or concept_data.get("boss_fight_logic", "why your scenario was wrong")
            )
            directive = (
                f"OVERRIDE FIREWALL: The user is frustrated and gave up. Take a deep "
                f"breath and gently explain exactly why your scenario was wrong in a full, "
                f"helpful paragraph based on this rule: '{ans}'. Make sure they really "
                f"understand the physics or logic. After explaining it, you MUST end your "
                f"response by asking a strict Yes or No question: 'Now that we know the "
                f"secret, do you want to try explaining it to me so we can finish?'"
            )

        else:
            directive = (
                "The user said something unexpected. Bring the conversation back "
                "to whether your scenario idea is actually correct."
            )

    # Reset clarification counter on any non-question turn
    if classification != "question":
        new_clarification = 0

    return directive, new_frustration, new_clarification


# ---------------------------------------------------------------------------
# E. SESSION STATE INITIALISATION
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialise all st.session_state keys on first load only."""
    defaults = {
        "messages":              [],
        "current_phase":         1,
        "frustration_counter":   0,
        "clarification_counter": 0,
        "boss_fight_attempts":   0,
        "concept_data":          None,
        "pips_last_question":    "",
        "awaiting_retry":        False,
        "session_ended":         False,
        "session_won":           False,
        "game_started":          False,
        "pending_levels":        [],   # AI-generated concepts awaiting parent review
        "current_concept_id":    None, # stable JSON key for the active concept
        "last_prompt":                "",     # debug: last prompt sent to Ollama
        "last_response":              "",     # debug: last raw response from Ollama
        "last_classification":        "None", # debug: last evaluator verdict
        "last_evaluator_rationale":   "None", # debug: raw evaluator JSON before parsing
        # audio
        "voice_enabled":   True,   # TTS: persona speaks its responses aloud
        "mic_enabled":     False,  # STT: show microphone recorder widget
        "last_spoken_idx": -1,     # TTS: index of last message already spoken (avoids replay on rerun)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Profile is loaded from disk once per browser session
    if "profile" not in st.session_state:
        st.session_state.profile = load_profile()


def start_session(concept_data: dict) -> None:
    """
    Reset all game state and seed Pip's opening line.
    Does NOT clear st.session_state.profile so the Trophy Room persists.
    """
    st.session_state.messages              = []
    st.session_state.current_phase         = 1
    st.session_state.frustration_counter   = 0
    st.session_state.clarification_counter = 0
    st.session_state.boss_fight_attempts   = 0
    st.session_state.concept_data          = concept_data
    st.session_state.pips_last_question    = concept_data["story_intro"]
    st.session_state.awaiting_retry        = False
    st.session_state.session_ended         = False
    st.session_state.session_won           = False
    st.session_state.game_started          = True
    st.session_state.last_spoken_idx       = -1  # reset TTS pointer for the new session

    st.session_state.messages.append({
        "role":    "assistant",
        "content": concept_data["story_intro"],
    })


# ---------------------------------------------------------------------------
# F. AUDIO HELPERS
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_whisper():
    """
    Load the faster-whisper base model once and cache it for the lifetime of
    the Streamlit server process.  Called lazily so the app starts instantly
    even if the model has not been downloaded yet.
    compute_type="int8" halves RAM usage on CPU with negligible accuracy loss.

    Corporate SSL note: huggingface_hub uses httpx internally, which does NOT
    respect the REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE env variables.
    We use huggingface_hub.configure_http_backend() to inject an httpx.Client
    with verify=False so the one-time model download works behind SSL-inspection
    proxies (common on corporate Cisco networks).
    This override is scoped to huggingface_hub only and does not affect Ollama.
    """
    if not FASTER_WHISPER_AVAILABLE:
        return None
    try:
        import httpx
        import huggingface_hub
        # Inject an SSL-verification-disabled httpx client for the model download
        huggingface_hub.configure_http_backend(
            backend_factory=lambda: httpx.Client(verify=False)
        )
    except (ImportError, AttributeError):
        # Older huggingface_hub (<0.23) — fall back to env-var approach
        os.environ["CURL_CA_BUNDLE"]     = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""

    try:
        return _WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as exc:
        st.warning(
            f"Could not load Whisper model: {exc}\n\n"
            "The model download failed. This is usually a corporate SSL proxy issue. "
            "If the problem persists, run this once in a terminal to pre-download the model:\n\n"
            "```\npython -c \"import ssl; ssl._create_default_https_context = "
            "ssl._create_unverified_context; "
            "from huggingface_hub import snapshot_download; "
            "snapshot_download('Systran/faster-whisper-base')\"\n```"
        )
        return None


def transcribe_audio(audio_bytes: bytes) -> str | None:
    """
    Transcribe raw WAV bytes to text using a local faster-whisper model.
    language="en" skips auto-detection to eliminate latency.
    Returns the transcribed string, or None on failure / empty audio.
    """
    if not audio_bytes or not FASTER_WHISPER_AVAILABLE:
        return None
    model = _load_whisper()
    if model is None:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        segments, _ = model.transcribe(tmp_path, beam_size=5, language="en")
        text = " ".join(s.text for s in segments).strip()
        return text or None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def speak_text(text: str, persona_name: str = "Pip") -> None:
    """
    Inject a tiny JS snippet that uses the browser's built-in Web Speech API
    (SpeechSynthesis) to read the persona's message aloud.
    Runs 100% client-side — no network calls, no Python audio dependencies.
    Each persona gets a distinct voice profile (pitch + rate).
    No-ops when voice_enabled is False.
    """
    if not st.session_state.get("voice_enabled", True):
        return

    voice_profiles = {
        "Pip":   {"pitch": 1.4, "rate": 0.85},
        "Alex":  {"pitch": 1.1, "rate": 1.0},
        "Riley": {"pitch": 0.9, "rate": 1.1},
    }
    profile = voice_profiles.get(persona_name, {"pitch": 1.0, "rate": 1.0})

    # Escape backticks and backslashes so the text is safe inside a JS template literal
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`")

    js = f"""
    <script>
      (function() {{
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance(`{safe_text}`);
        u.pitch = {profile['pitch']};
        u.rate  = {profile['rate']};
        window.speechSynthesis.speak(u);
      }})();
    </script>
    """
    st.components.v1.html(js, height=0)


# ---------------------------------------------------------------------------
# G. CHAT HISTORY RENDERER
# ---------------------------------------------------------------------------

def get_persona_avatar() -> str:
    """
    Return the current persona's avatar emoji.
    Falls back to 👧🏼 (Pip) for legacy concepts that pre-date the persona system
    or when no game has started yet.
    """
    concept_data = st.session_state.get("concept_data")
    if concept_data:
        return (
            concept_data
            .get("persona_config", {})
            .get("avatar_emoji", "👧🏼")
        )
    return "👧🏼"


def render_chat_history(persona_avatar: str = "👧🏼", persona_name: str = "Pip") -> None:
    """
    Render all messages in st.session_state.messages.
      "assistant" → persona avatar passed in from the active concept's persona_config
      "user"      → student (avatar 👤)
      "system"    → phase/state banners as styled st.info boxes

    TTS: after rendering, speak the latest assistant message if it has not been
    spoken yet (guarded by last_spoken_idx to prevent re-reading on every rerun).
    """
    last_spoken_idx = st.session_state.get("last_spoken_idx", -1)
    latest_assistant_idx = -1
    latest_assistant_text = ""

    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "system":
            st.info(message["content"], icon="⚔️")
        else:
            avatar = persona_avatar if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
            if message["role"] == "assistant":
                latest_assistant_idx  = idx
                latest_assistant_text = message["content"]

    # Speak the latest assistant message only if it is new since the last render
    if (
        latest_assistant_idx > last_spoken_idx
        and latest_assistant_text
        and st.session_state.get("voice_enabled", True)
    ):
        speak_text(latest_assistant_text, persona_name)
        st.session_state.last_spoken_idx = latest_assistant_idx


# ---------------------------------------------------------------------------
# G. PARENT DASHBOARD RENDERER
# ---------------------------------------------------------------------------

def render_dashboard(knowledge: dict) -> None:
    """Render the Parent Insights & Analytics view."""
    student_name = st.session_state.profile.get("student_name", "Explorer")
    achievements = st.session_state.profile.get("achievements", {})

    st.title("📊 Parent Insights & Analytics")
    st.markdown(f"*Tracking progress for **{student_name}***")
    st.divider()

    # ── Summary Metrics ──────────────────────────────────────────────────────
    if achievements:
        total_mastered = len(achievements)
        total_friction = sum(v.get("frustration_triggers", 0) for v in achievements.values())

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏆 Concepts Mastered", total_mastered)
        with col2:
            st.metric(
                "😤 Learning Friction",
                total_friction,
                help="Total frustration triggers accumulated across all subjects.",
            )

        st.divider()

        # ── Progress Table ────────────────────────────────────────────────────
        st.subheader("📋 Concept Breakdown")
        concepts_map = knowledge.get("concepts", {})
        table_rows = [
            {
                "Concept":              concepts_map.get(cid, {}).get("name", cid),
                "Status":               data.get("status", "Unknown"),
                "Frustration Triggers": data.get("frustration_triggers", 0),
                "Boss Fight Attempts":  data.get("boss_fight_attempts", 0),
            }
            for cid, data in achievements.items()
        ]
        st.dataframe(table_rows, use_container_width=True)

        st.divider()

        # ── Pedagogical Advice ────────────────────────────────────────────────
        st.subheader("💡 Suggested Home Activity")

        hardest_id, hardest_stats = max(
            achievements.items(),
            key=lambda kv: kv[1].get("frustration_triggers", 0),
        )
        friction     = hardest_stats.get("frustration_triggers", 0)
        hardest_data = knowledge.get("concepts", {}).get(hardest_id, {})
        hardest_name = hardest_data.get("name", hardest_id)
        activity     = hardest_data.get(
            "home_activity",
            f"Review '{hardest_name}' by looking for real-world examples around the house!",
        )

        if friction > 0:
            st.info(
                f"**High friction on '{hardest_name}'** — {friction} struggle moment(s) detected.\n\n"
                f"💡 *Try this at home:* {activity}",
                icon="🏠",
            )
        else:
            st.success(
                f"**'{hardest_name}'** was mastered smoothly with zero frustration triggers. "
                "Fantastic work! Keep the momentum going with a new concept.",
                icon="🌟",
            )

        st.divider()

    else:
        st.info(
            "No concepts mastered yet!\n\n"
            "Switch to **🎮 Play (Kid Mode)**, pick a concept, and complete a full session "
            "to see analytics appear here.",
            icon="🌱",
        )

    # ── Review Queue (ChromaDB) ───────────────────────────────────────────────
    try:
        _collection   = get_chroma_collection()
        pending_results = _collection.get(
            where   = {"status": "pending"},
            include = ["metadatas"],
        )
        pending_ids       = pending_results.get("ids", [])
        pending_metadatas = pending_results.get("metadatas", [])
    except Exception as exc:
        st.warning(f"⚠️ Could not query ChromaDB review queue: {exc}")
        pending_ids       = []
        pending_metadatas = []

    if pending_ids:
        st.subheader(f"🗂️ Concept Review Queue ({len(pending_ids)} pending)")
        st.caption("Review each AI-generated concept before it enters the active curriculum.")

        # ── Approve All ──────────────────────────────────────────────────────
        if st.button("✅ Approve All Pending Levels", use_container_width=True):
            collection = get_chroma_collection()
            for item_id, meta in zip(pending_ids, pending_metadatas):
                collection.update(
                    ids       = [item_id],
                    metadatas = [{**meta, "status": "approved"}],
                )
            st.success(f"✅ {len(pending_ids)} concept(s) approved and added to the curriculum!")
            st.rerun()

        # ── Per-concept cards ────────────────────────────────────────────────
        for i, (item_id, meta) in enumerate(zip(pending_ids, pending_metadatas)):
            try:
                concept = json.loads(meta.get("concept_json", "{}"))
            except json.JSONDecodeError:
                concept = {}

            concept_name = meta.get("concept_name") or concept.get("name", "Untitled Concept")
            persona_tag  = (
                f"  ·  {meta['persona_name']} ({meta['complexity_level']})"
                if meta.get("persona_name") else ""
            )
            subject_tag  = meta.get("subject") or concept.get("subject", "General")

            with st.expander(f"📖 {concept_name}{persona_tag}", expanded=True):
                st.write(f"**Subject:** `{subject_tag}`")
                st.markdown(f"**Opening Puzzle:**\n> {concept.get('story_intro', '—')}")
                st.markdown(f"**Boss Fight Scenario:**\n> {concept.get('verification_scenario', '—')}")
                if concept.get("home_activity"):
                    st.markdown(f"**Home Activity:**\n> {concept['home_activity']}")

                approve_col, reject_col = st.columns([1, 2])

                with approve_col:
                    if st.button("✅ Approve", key=f"approve_{item_id}",
                                 use_container_width=True):
                        get_chroma_collection().update(
                            ids       = [item_id],
                            metadatas = [{**meta, "status": "approved"}],
                        )
                        st.success(f"✅ '{concept_name}' approved!")
                        st.rerun()

                with reject_col:
                    reject_reason = st.selectbox(
                        "Reason for Rejection",
                        [
                            "Hallucination / Factually Wrong",
                            "Too Complex for Age Group",
                            "Formatting / JSON Error",
                            "Off-Topic or Irrelevant",
                            "Other",
                        ],
                        key=f"reason_{item_id}",
                        label_visibility="collapsed",
                    )
                    if st.button("🗑️ Reject & Delete", key=f"reject_{item_id}",
                                 use_container_width=True):
                        # 1. Write to the Dead Letter Queue file FIRST (before delete)
                        write_dlq(item_id, concept, reject_reason)
                        # 2. Remove the vector from ChromaDB entirely so it
                        #    no longer pollutes the embedding space.
                        get_chroma_collection().delete(ids=[item_id])
                        st.warning(
                            f"Concept **'{concept_name}'** rejected, logged to "
                            f"`{DLQ_FILE}` for developer review, and purged from database.",
                            icon="🗑️",
                        )
                        st.rerun()

        st.divider()

    # ── Active Curriculum ─────────────────────────────────────────────────────
    st.subheader("📋 Active Curriculum")
    st.caption("All approved concepts currently in the game. Remove any that are outdated or incorrect.")
    try:
        _col = get_chroma_collection()
        active_results = _col.get(
            where   = {"status": "approved"},
            include = ["metadatas"],
        )
        active_ids   = active_results.get("ids", [])
        active_metas = active_results.get("metadatas", [])
    except Exception as exc:
        st.warning(f"⚠️ Could not load active curriculum: {exc}")
        active_ids, active_metas = [], []

    if active_ids:
        # Sort by subject then sequence_order for a clean display
        active_sorted = sorted(
            zip(active_ids, active_metas),
            key=lambda x: (x[1].get("subject", ""), float(x[1].get("sequence_order", 999))),
        )
        for item_id, meta in active_sorted:
            seq          = meta.get("sequence_order", "—")
            concept_name = meta.get("concept_name", item_id)
            subject      = meta.get("subject", "General")
            col1, col2   = st.columns([5, 1])
            with col1:
                st.write(f"**{seq}** · {concept_name} `{subject}`")
            with col2:
                if st.button("🗑️", key=f"del_{item_id}", help=f"Remove '{concept_name}'"):
                    # Preserve the concept name in the student profile before deletion
                    # so Trophy Room entries remain human-readable
                    profile = st.session_state.profile
                    for ach_id, ach_data in profile.get("achievements", {}).items():
                        if ach_id == item_id and "concept_name" not in ach_data:
                            ach_data["concept_name"] = concept_name
                    save_profile(profile)
                    _col.delete(ids=[item_id])
                    st.toast(f"'{concept_name}' removed from active curriculum.")
                    st.rerun()
    else:
        st.info("No approved concepts yet. Approve concepts from the Review Queue above.", icon="📭")

    st.divider()

    # ── Sequence Repair ───────────────────────────────────────────────────────
    with st.expander("🔧 Repair Sequence Numbers", expanded=False):
        st.caption(
            "Concepts uploaded before sequencing was introduced have "
            "sequence_order=999. Click below to assign real numbers based on "
            "the current order within each subject."
        )
        if st.button("🔢 Assign Sequence Numbers to All Concepts",
                     use_container_width=True):
            try:
                col     = get_chroma_collection()
                all_res = col.get(include=["metadatas"])
                all_ids = all_res.get("ids", [])
                all_metas = all_res.get("metadatas", [])

                # Group by subject, then assign 1-based order
                by_subject: dict = {}
                for cid, meta in zip(all_ids, all_metas):
                    subj = meta.get("subject", "General")
                    by_subject.setdefault(subj, []).append((cid, meta))

                updated = 0
                for subj, entries in by_subject.items():
                    for seq_num, (cid, meta) in enumerate(entries, start=1):
                        if int(meta.get("sequence_order", 999)) == 999:
                            col.update(
                                ids       = [cid],
                                metadatas = [{**meta, "sequence_order": seq_num}],
                            )
                            updated += 1

                if updated:
                    st.success(f"✅ Assigned sequence numbers to {updated} concept(s). Refresh the page.")
                else:
                    st.info("All concepts already have sequence numbers assigned.")
            except Exception as exc:
                st.error(f"Repair failed: {exc}")

    st.divider()

    # ── PDF Ingestion ─────────────────────────────────────────────────────────
    st.subheader("📄 Ingest New Concept from PDF")
    st.caption(
        "Upload a textbook page, article, or worksheet. "
        "The AI will design a new Pip puzzle from it for your review."
    )

    st.info("""
    **🚀 Optimization Notice:**  
    GemmaGenius performs best with **text-heavy, digitally-born PDFs** (e.g., OpenStax, textbooks, or clean export files). 
    Scanned documents or complex image-heavy layouts may result in lower-quality logic traps.
    """, icon="📄")

    selected_subject = st.selectbox(
        "Assign Subject",
        ["General", "Mathematics", "Science", "Biology", "Finance", "Reading", "History", "Technology"],
        help="This tag will be stored with every concept extracted from the PDF.",
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("🔍 Generate Concept from PDF", type="primary"):
            with st.spinner("Extracting text from PDF…"):
                extracted_text = extract_pdf_text(uploaded_file)

            # ── Fail-fast guardrail ───────────────────────────────────────────
            if len(extracted_text) < PDF_MIN_CHARS:
                st.warning(
                    "This PDF is too short or poor quality for AI processing. "
                    "Please use a clean text version.",
                    icon="⚠️",
                )
                st.stop()

            with st.spinner("AI is designing puzzles from the PDF…"):
                sys_p, usr_p = compile_concept_generation_prompt(extracted_text)
                raw_json     = call_ollama(sys_p, usr_p, json_mode=True)

            # Strip markdown fences the model sometimes adds even in JSON mode
            clean_json = raw_json.strip()
            if clean_json.startswith("```"):
                # Remove opening fence (```json or ```)
                clean_json = clean_json.split("\n", 1)[-1]
            if clean_json.endswith("```"):
                clean_json = clean_json.rsplit("```", 1)[0]
            clean_json = clean_json.strip()

            try:
                parsed = json.loads(clean_json)
            except json.JSONDecodeError as exc:
                st.error(
                    f"The AI returned malformed JSON ({exc}). "
                    "Try uploading a cleaner PDF or a shorter excerpt.",
                    icon="❌",
                )
                with st.expander("🛠️ Raw LLM output (for debugging)"):
                    st.code(raw_json[:3000], language="json")
                st.stop()

            # Handle {skip: true} response
            if parsed.get("skip"):
                st.warning("The AI could not find enough content to create a concept. Try a richer PDF.")
                st.stop()

            # Support multi-concept array envelope OR legacy single-object fallback
            concepts_to_embed = parsed.get("extracted_concepts")
            if not concepts_to_embed:
                # Legacy: bare single object
                concepts_to_embed = [parsed]

            # Determine the next sequence number for this subject in ChromaDB
            try:
                _existing = get_chroma_collection().get(
                    where   = {"subject": selected_subject},
                    include = ["metadatas"],
                )
                existing_seqs = [
                    int(m.get("sequence_order", 0))
                    for m in _existing.get("metadatas", [])
                    if m.get("sequence_order") is not None
                ]
                next_seq = max(existing_seqs, default=0) + 1
            except Exception:
                next_seq = 1

            embedded_count = 0
            failed_names   = []
            for concept in concepts_to_embed:
                concept.setdefault("concept_name", concept.get("name", "Untitled Concept"))
                concept.setdefault("name", concept["concept_name"])
                concept["subject"]        = selected_subject   # stamp subject chosen by parent
                concept["sequence_order"] = next_seq           # stamp sequential order
                next_seq += 1
                with st.spinner(f"Embedding **'{concept['concept_name']}'**…"):
                    ok = _embed_and_upsert_concept(concept, status="pending")
                if ok:
                    embedded_count += 1
                else:
                    failed_names.append(concept["concept_name"])

            if embedded_count:
                st.success(
                    f"✅ **{embedded_count} concept(s)** added to the Review Queue. "
                    "Scroll up to preview and approve them."
                )
            if failed_names:
                st.error(
                    f"Could not embed: {', '.join(failed_names)}. "
                    "Make sure Ollama is running and `nomic-embed-text` is pulled.",
                    icon="❌",
                )
            if not embedded_count:
                st.stop()
            st.rerun()


# ---------------------------------------------------------------------------
# H. MAIN APP
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="GemmaGenius",
        page_icon="🧠",
        layout="wide",
    )

    init_session_state()

    # ── Load knowledge graph (legacy hand-authored concepts from knowledge.json) ─
    try:
        with open("knowledge.json", "r", encoding="utf-8") as fh:
            knowledge = json.load(fh)
    except FileNotFoundError:
        knowledge = {"concepts": {}}
    except json.JSONDecodeError:
        knowledge = {"concepts": {}}

    legacy_concepts = knowledge.get("concepts", {})

    # ── Load approved concepts from ChromaDB ─────────────────────────────────
    chroma_concepts = {}   # key -> concept dict
    try:
        _col = get_chroma_collection()
        approved = _col.get(
            where   = {"status": "approved"},
            include = ["metadatas"],
        )
        for cid, meta in zip(approved.get("ids", []), approved.get("metadatas", [])):
            try:
                concept_obj = json.loads(meta.get("concept_json", "{}"))
            except json.JSONDecodeError:
                concept_obj = {}
            # Backfill flat metadata fields that may not be inside concept_json
            concept_obj.setdefault("name", meta.get("concept_name", cid))
            concept_obj.setdefault("subject", meta.get("subject", "General"))
            concept_obj.setdefault("sequence_order", meta.get("sequence_order", 999))
            chroma_concepts[cid] = concept_obj
    except Exception as exc:
        st.warning(f"⚠️ Could not load concepts from ChromaDB: {exc}", icon="⚠️")

    # Merge: ChromaDB approved concepts take precedence; legacy fills the rest
    concepts = {**legacy_concepts, **chroma_concepts}
    concept_keys  = list(concepts.keys())
    concept_names = [concepts[k]["name"] for k in concept_keys]

    # ── Build subject-grouped index for the sidebar folder UI ─────────────────
    # ChromaDB concepts carry "subject" and "sequence_order" fields.
    # Legacy hand-authored concepts default to "Classic" subject, order 999.
    grouped_concepts: dict[str, list[tuple[str, dict]]] = {}
    for key in concept_keys:
        concept_obj = concepts[key]
        subject = concept_obj.get("subject", "Classic" if key in legacy_concepts else "General")
        grouped_concepts.setdefault(subject, []).append((key, concept_obj))

    # Sort concepts within each subject by their sequence_order
    for subject in grouped_concepts:
        grouped_concepts[subject].sort(key=lambda x: int(x[1].get("sequence_order", 999)))

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🧠 GemmaGenius")
        st.markdown(f"*Hello, **{st.session_state.profile['student_name']}**!*")
        st.divider()

        # Navigation toggle — switching views does NOT clear messages
        nav_view = st.radio(
            "Navigation",
            ["🎮 Play (Kid Mode)", "📊 Dashboard (Parent Mode)"],
            key="nav_view",
        )
        st.divider()

        # Trophy Room — always visible for motivation
        achievements = st.session_state.profile.get("achievements", {})
        if achievements:
            st.markdown("### 🏆 Trophy Room")
            for cid in achievements:
                display = concepts.get(cid, {}).get("name", cid)
                st.markdown(f"⭐ **{display}**")
            st.divider()

        # Game controls are only meaningful in Play mode
        if nav_view == "🎮 Play (Kid Mode)":
            if grouped_concepts:
                st.subheader("📚 My Subjects")
                active_id    = st.session_state.get("current_concept_id")
                achievements = st.session_state.profile.get("achievements", {})

                LOOK_AHEAD = 3  # next 1 immediate + 2 exploratory levels

                for subject, items in sorted(grouped_concepts.items()):
                    is_active_folder = any(cid == active_id for cid, _ in items)
                    with st.expander(f"📁 {subject} ({len(items)})", expanded=is_active_folder):

                        # ── Pass 1: find the highest mastered sequence in this folder ──
                        max_mastered_seq = 0
                        for cid, concept_obj in items:
                            if cid in achievements:
                                seq = int(concept_obj.get("sequence_order", 0))
                                if seq > max_mastered_seq:
                                    max_mastered_seq = seq

                        # ── Pass 2: render with look-ahead buffer ──────────────────────
                        for cid, concept_obj in items:
                            concept_name = concept_obj.get("name", cid)
                            seq          = int(concept_obj.get("sequence_order", 999))
                            is_mastered  = cid in achievements
                            is_playing   = cid == active_id

                            # seq=999 means the concept predates sequencing — treat as unlocked
                            # so legacy / pre-migration concepts are always accessible
                            is_unlocked = (
                                is_mastered
                                or seq == 999
                                or seq <= max_mastered_seq + LOOK_AHEAD
                            )

                            if is_unlocked:
                                if is_mastered:
                                    icon = "⭐"
                                elif is_playing:
                                    icon = "▶"
                                else:
                                    icon = "○"

                                if st.button(
                                    f"{icon} {concept_name}",
                                    key=f"nav_{cid}",
                                    use_container_width=True,
                                ):
                                    st.session_state.current_concept_id = cid

                                    # Decide whether JIT generation is needed.
                                    # Legacy knowledge.json concepts already have story_intro;
                                    # ChromaDB skeleton concepts (JIT) do not.
                                    has_story = bool(concept_obj.get("story_intro"))

                                    if has_story:
                                        # Legacy path — use concept as-is (Scenario Polymorphism)
                                        concept_data = concept_obj.copy()
                                        if "variants" in concept_data:
                                            variant = random.choice(concept_data["variants"])
                                            concept_data["story_intro"]               = variant["story_intro"]
                                            concept_data["verification_scenario"]     = variant["verification_scenario"]
                                            concept_data["verification_ground_truth"] = variant["verification_ground_truth"]
                                        start_session(concept_data)
                                        st.rerun()
                                    else:
                                        # JIT path — generate story + boss fight on the fly
                                        ground_truth = concept_obj.get("ground_truth_logic", "")
                                        concept_data = generate_level_jit(
                                            concept_name, ground_truth, skeleton=concept_obj
                                        )
                                        if concept_data:
                                            start_session(concept_data)
                                            st.rerun()
                                        else:
                                            st.error(
                                                "Pip got confused generating this puzzle! "
                                                "Try clicking the level again.",
                                                icon="❌",
                                            )
                            else:
                                st.button(
                                    f"🔒 {concept_name}",
                                    key=f"lock_{cid}",
                                    use_container_width=True,
                                    disabled=True,
                                )
            else:
                st.info("No concepts approved yet. Ask a parent to add some in the Dashboard!", icon="📖")

            st.divider()

            if st.session_state.game_started and st.session_state.concept_data:
                st.markdown(f"**Concept:** {st.session_state.concept_data['name']}")
                phase_label = (
                    "Phase 1 — Elicitation"
                    if st.session_state.current_phase == 1
                    else "Phase 2 — Boss Fight ⚔️"
                )
                st.markdown(f"**Phase:** {phase_label}")
                st.markdown(f"**Misses:** {st.session_state.frustration_counter}")

        st.divider()
        st.caption("Running on local Ollama · No data leaves your device")

        with st.expander("🔊 Audio Settings"):
            # value= is intentionally omitted — init_session_state() sets the defaults
            # and Streamlit warns if both value= and session_state[key] are specified
            st.checkbox(
                "Speak persona responses",
                key="voice_enabled",
                help="The persona reads its messages aloud using your browser's built-in voice engine.",
            )
            mic_available = AUDIO_RECORDER_AVAILABLE and FASTER_WHISPER_AVAILABLE
            mic_help = (
                "Record your answer with the microphone instead of typing."
                if mic_available
                else "Install audio-recorder-streamlit and faster-whisper to enable mic input."
            )
            st.checkbox(
                "Mic input (speak your answer)",
                key="mic_enabled",
                disabled=not mic_available,
                help=mic_help,
            )

        with st.expander("🔧 Developer Settings"):
            debug_mode = st.checkbox("🐛 Enable Debug Mode", value=False, key="debug_mode")

    # ════════════════════════════════════════════════════════════════════════
    # MAIN AREA — conditional on navigation selection
    # ════════════════════════════════════════════════════════════════════════

    if nav_view == "📊 Dashboard (Parent Mode)":
        render_dashboard(knowledge)
        st.stop()

    # ── Play (Kid Mode) ───────────────────────────────────────────────────────
    # Resolve persona from the active concept; fall back to Pip for legacy concepts
    _concept      = st.session_state.get("concept_data") or {}
    _persona      = _concept.get("persona_config", {})
    persona_name  = _persona.get("name", "Pip")
    persona_avatar = _persona.get("avatar_emoji", "👧🏼")

    st.title("GemmaGenius")
    st.caption(
        f"Explain the concept to {persona_name} — "
        "they learn by asking questions, not by being told."
    )

    if not st.session_state.game_started:
        st.info("👈 Choose a concept in the sidebar and click **Start / Reset Puzzle** to begin.")
        st.stop()

    # ── Render full chat history ──────────────────────────────────────────────
    render_chat_history(persona_avatar, persona_name)

    # ── Win state: show celebration and block further input ───────────────────
    if st.session_state.session_won:
        st.balloons()
        st.success(f"⭐ TRUE MASTERY ACHIEVED! You explained the concept AND caught {persona_name}'s mistake!")
        st.info("Choose a concept in the sidebar and click **Start / Reset Puzzle** to play again!")
        st.stop()

    # ── Session ended (give_up declined, not a win) ───────────────────────────
    if st.session_state.session_ended:
        st.info(
            "Session ended. Choose a concept in the sidebar and "
            "click **Start / Reset Puzzle** to play again!"
        )
        st.stop()

    # ── Give-up recovery: replace chat input with Yes / No buttons ───────────
    if st.session_state.awaiting_retry:
        st.markdown("---")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Yes, let's try again!", use_container_width=True):
                st.session_state.awaiting_retry        = False
                st.session_state.frustration_counter   = 0
                st.session_state.clarification_counter = 0

                concept_data = st.session_state.concept_data
                opening = (
                    concept_data["story_intro"]
                    if st.session_state.current_phase == 1
                    else concept_data["verification_scenario"]
                )

                st.session_state.pips_last_question = opening
                st.session_state.messages.append({
                    "role":    "system",
                    "content": f"⚙️ {persona_name} resets the puzzle... Let's try again!",
                })
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": opening,
                })
                st.rerun()
        with col2:
            if st.button("❌ No, I'm done", use_container_width=True):
                st.session_state.awaiting_retry  = False
                st.session_state.session_ended   = True
                st.session_state.messages.append({
                    "role":    "system",
                    "content": "Session ended gracefully. You did a great job trying! 🌟",
                })
                st.rerun()
        st.stop()

    # ── Chat input (text + optional mic) ──────────────────────────────────────
    user_input: str | None = None

    if st.session_state.get("mic_enabled") and AUDIO_RECORDER_AVAILABLE:
        # Side-by-side layout: wide text input + narrow mic button
        input_col, mic_col = st.columns([6, 1])
        with input_col:
            user_text = st.chat_input(f"Explain it to {persona_name}...")
        with mic_col:
            audio_bytes = _audio_recorder(
                text="",
                recording_color="#e84118",
                neutral_color="#353b48",
                icon_size="2x",
                key="mic_recorder",
            )
        # Typed input wins; fall back to transcribed voice
        if user_text:
            user_input = user_text
        elif audio_bytes:
            with st.spinner("Listening..."):
                user_input = transcribe_audio(audio_bytes)
            if user_input:
                st.caption(f"Heard: *{user_input}*")
    else:
        user_input = st.chat_input(f"Explain it to {persona_name}...")

    if user_input:

        # 1. Append and immediately display the user's message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        concept_data = st.session_state.concept_data

        # Track each student submission during the Boss Fight
        if st.session_state.current_phase == 2:
            st.session_state.boss_fight_attempts += 1

        with st.spinner(f"{persona_name} is thinking..."):

            # 2. Call The Evaluator (phase-aware)
            eval_sys, eval_usr = compile_evaluator_prompt(
                concept_data,
                latest_user_input  = user_input,
                pips_last_question = st.session_state.pips_last_question,
                phase              = st.session_state.current_phase,
            )
            raw_verdict = call_ollama(eval_sys, eval_usr, json_mode=True)

            # Save the raw evaluator response before any parsing for the debug console
            st.session_state.last_evaluator_rationale = raw_verdict

            try:
                classification = (
                    json.loads(raw_verdict)
                    .get("classification", "miss")
                    .lower()
                    .strip()
                )
            except (json.JSONDecodeError, AttributeError):
                classification = "miss"

            # Save the parsed verdict so the debug console always shows the latest
            st.session_state.last_classification = classification.upper()

            # ── Phase 2 Mastery → TRUE WIN STATE ─────────────────────────────
            if st.session_state.current_phase == 2 and classification == "mastery":

                # Persist the achievement keyed by the stable JSON concept ID
                concept_id = st.session_state.get(
                    "current_concept_id",
                    concept_data.get("name", "Unknown_Concept"),
                )
                st.session_state.profile["achievements"][concept_id] = {
                    "status":               "Mastered",
                    "frustration_triggers": st.session_state.frustration_counter,
                    "boss_fight_attempts":  st.session_state.boss_fight_attempts,
                }
                save_profile(st.session_state.profile)

                win_msg = (
                    "⭐ **TRUE MASTERY ACHIEVED!** ⭐\n\n"
                    "You explained the concept AND caught Pip's mistake!\n\n"
                    "You explained it so well, even an 8-year-old gets it now. You're a genius! 🎉"
                )
                st.session_state.messages.append({"role": "assistant", "content": win_msg})
                st.session_state.session_ended = True
                st.session_state.session_won   = True
                st.balloons()
                st.rerun()

            # ── Phase 1 Mastery → seamless transition via a single Pip turn ──
            elif st.session_state.current_phase == 1 and classification == "mastery":
                st.session_state.current_phase         = 2
                st.session_state.frustration_counter   = 0
                st.session_state.clarification_counter = 0

                st.session_state.messages.append({
                    "role":    "system",
                    "content": "Phase 1 complete! Boss Fight starting...",
                })

                boss_scenario = concept_data.get("verification_scenario", "my new scenario")
                directive = (
                    f"OVERRIDE FIREWALL: The user just perfectly explained the concept! "
                    f"Enthusiastically validate how smart they are. Then, IMMEDIATELY present "
                    f"this new scenario and ask if your logic is correct: '{boss_scenario}'"
                )

                pip_system   = compile_pip_prompt(concept_data, directive)
                pip_response = call_ollama(pip_system, user_prompt=user_input, json_mode=False)
                if not pip_response:
                    pip_response = f"[System] {persona_name} could not respond. Please try again."

                st.session_state.messages.append({"role": "assistant", "content": pip_response})
                st.session_state.pips_last_question = pip_response
                st.rerun()

            # ── All other classifications: route → directive → call Pip ──────
            else:
                directive, new_f, new_c = build_state_directive(
                    classification        = classification,
                    concept_data          = concept_data,
                    current_phase         = st.session_state.current_phase,
                    frustration_counter   = st.session_state.frustration_counter,
                    clarification_counter = st.session_state.clarification_counter,
                    latest_user_input     = user_input,
                )
                st.session_state.frustration_counter   = new_f
                st.session_state.clarification_counter = new_c

                pip_system   = compile_pip_prompt(concept_data, directive)
                pip_response = call_ollama(pip_system, user_prompt=user_input, json_mode=False)

                if not pip_response:
                    pip_response = f"[System] {persona_name} could not respond. Please try again."

                st.session_state.messages.append({"role": "assistant", "content": pip_response})
                st.session_state.pips_last_question = pip_response

                if classification == "give_up":
                    st.session_state.awaiting_retry = True

                st.rerun()

    # ── Developer Console ─────────────────────────────────────────────────────
    # Rendered outside the chat input block so it persists at the bottom of the
    # Play view even when no new message is being processed.
    if st.session_state.get("debug_mode"):
        with st.expander("🛠️ Developer Console: Telemetry", expanded=True):

            # ── Evaluator Telemetry ───────────────────────────────────────────
            st.write("### 🧠 Background Evaluator")
            st.info(f"**Classification:** {st.session_state.get('last_classification', 'N/A')}")
            st.write("**Evaluator Rationale:**")
            st.code(
                st.session_state.get("last_evaluator_rationale", "N/A"),
                language="markdown",
            )

            st.divider()

            # ── Pip Telemetry ─────────────────────────────────────────────────
            st.write("### 👧🏼 Pip Generation")
            if st.session_state.get("last_prompt"):
                st.write("**Raw Prompt Sent to Pip:**")
                st.code(st.session_state.last_prompt, language="markdown")
                st.write("**Raw Output from Pip:**")
                st.code(st.session_state.last_response, language="json")

            st.divider()

            # ── Session State Dump ────────────────────────────────────────────
            st.write("### 💾 Session State")
            st.json({
                k: v
                for k, v in st.session_state.items()
                if k not in (
                    "messages",
                    "last_prompt",
                    "last_response",
                    "last_evaluator_rationale",
                )
            })


if __name__ == "__main__":
    main()
