"""
GemmaGenius — Streamlit Web UI  (Pillar A: Persistence & Parent Dashboard)
==========================================================================
Dual-view navigation: Kid Mode (chat) and Parent Dashboard (analytics).
Student progress is persisted to student_profile.json between sessions.

Run with:
    streamlit run app.py
"""

import io
import json
import os
import random
import requests
import streamlit as st
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL_NAME     = "gemma4:e4b"
TIMEOUT_SECS   = 120
PROFILE_FILE   = "student_profile.json"
KNOWLEDGE_FILE = "knowledge.json"
PDF_MIN_CHARS  = 500   # fail-fast guardrail threshold

# Concept-specific home activity suggestions for the Parent Dashboard
CONCEPT_ACTIVITIES: dict[str, str] = {
    "The Book Corner": (
        "Try a 'corner hunt' at home — find 5 objects with perfect L-shaped corners "
        "(book spine, door frame, table edge) and trace each one with a finger."
    ),
    "The Earth's Invisible Tug": (
        "Do a 'heavy vs light drop' experiment — hold a pencil and a crumpled paper ball "
        "at the same height, then release both simultaneously. Talk about what you notice!"
    ),
}


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
# PDF INGESTION HELPERS
# ---------------------------------------------------------------------------

def extract_pdf_text(uploaded_file) -> str:
    """Return all text from an uploaded PDF file object."""
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def compile_concept_generation_prompt(extracted_text: str) -> tuple[str, str]:
    """Build the (system, user) prompt to turn raw PDF text into a game concept."""
    system_prompt = """\
You are an educational content designer who creates puzzles for children aged 6-12.
Given source material, extract ONE key concept and design a complete GemmaGenius puzzle.

OUTPUT RULES — respond with ONLY valid JSON, nothing else:
{
  "name":                     "Short imaginative title (5 words or fewer)",
  "secret_fact":              "One sentence: the core mechanism in plain language",
  "goal":                     "What the student must understand to win",
  "story_intro":              "Pip's opening puzzle (2-3 sentences as a confused 8-year-old, ends with a question)",
  "story_bridge":             "Pip's celebration when Phase 1 is solved (1-2 sentences)",
  "evaluator_ground_truth":   "Phase 1 ground truth: what the student must explain",
  "verification_scenario":    "Pip's flawed Phase 2 scenario (2-3 sentences with a logical mistake, ends with 'right?')",
  "verification_ground_truth":"Phase 2 ground truth: exactly why Pip's scenario is wrong"
}

STYLE RULES:
- story_intro and verification_scenario must sound like a real confused 8-year-old, not a textbook.
- Keep all text child-friendly and concrete — use physical objects and everyday situations.
- verification_scenario must contain a clear, correctable misconception about the concept.
"""
    user_prompt = (
        f"SOURCE MATERIAL:\n{extracted_text[:8000]}\n\n"
        "Generate the concept puzzle JSON:"
    )
    return system_prompt, user_prompt


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
    ground_truth = (
        concept_data["verification_ground_truth"]
        if phase == 2
        else concept_data["evaluator_ground_truth"]
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
  "question"    — Student is asking a clarifying question about the concept or the puzzle.
  "off_topic"   — Student is joking, typing nonsense, or talking about something completely unrelated to the current task.
  "give_up"     — Student explicitly says "I don't know", "tell me", "you explain it", or expresses deep frustration and a desire to quit the puzzle.

Be strict but fair. Do not guess intent — classify what was actually said.
IMPORTANT: Only use "give_up" when the student clearly and explicitly surrenders. A wrong answer is still a "miss", not a "give_up".
"""

    user_prompt = (
        f"PIP'S LAST QUESTION:\n{pips_last_question}\n\n"
        f"STUDENT'S LATEST RESPONSE:\n{latest_user_input}\n\n"
        "Your JSON classification:"
    )

    return system_prompt, user_prompt


def compile_pip_prompt(concept_data: dict, state_directive: str) -> str:
    """Build Pip's system prompt, injecting a dynamic state directive each turn."""
    return f"""You are Pip, an enthusiastic 8-year-old trying to solve a puzzle.
The context of your puzzle is based on the initial story you were told.
You are talking to a friend who is helping you figure it out.

INFORMATION FIREWALL — THIS IS YOUR MOST IMPORTANT RULE:
CRITICAL: Unless your specific State Directive explicitly tells you to give the answer,
you must NEVER give away the exact answer or vocabulary word. Wait for the user to explain it to you.

SPEAKING RULES:
- Maximum 3 short, simple sentences per reply.
- Use the language of a real 8-year-old. No textbook words.
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
            analogy = concept_data.get("secret_fact", "the core concept")
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
            ans = concept_data.get("verification_ground_truth", "why your scenario was wrong")
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
        "last_prompt":                "",     # debug: last prompt sent to Ollama
        "last_response":              "",     # debug: last raw response from Ollama
        "last_classification":        "None", # debug: last evaluator verdict
        "last_evaluator_rationale":   "None", # debug: raw evaluator JSON before parsing
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

    st.session_state.messages.append({
        "role":    "assistant",
        "content": concept_data["story_intro"],
    })


# ---------------------------------------------------------------------------
# F. CHAT HISTORY RENDERER
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


def render_chat_history() -> None:
    """
    Render all messages in st.session_state.messages.
      "assistant" → persona avatar (dynamic, from persona_config or fallback 👧🏼)
      "user"      → student (avatar 👤)
      "system"    → phase/state banners as styled st.info boxes
    """
    for message in st.session_state.messages:
        if message["role"] == "system":
            st.info(message["content"], icon="⚔️")
        else:
            avatar = get_persona_avatar() if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])


# ---------------------------------------------------------------------------
# G. PARENT DASHBOARD RENDERER
# ---------------------------------------------------------------------------

def render_dashboard() -> None:
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
        table_rows = [
            {
                "Concept":              concept_name,
                "Status":               data.get("status", "Unknown"),
                "Frustration Triggers": data.get("frustration_triggers", 0),
                "Boss Fight Attempts":  data.get("boss_fight_attempts", 0),
            }
            for concept_name, data in achievements.items()
        ]
        st.dataframe(table_rows, use_container_width=True)

        st.divider()

        # ── Pedagogical Advice ────────────────────────────────────────────────
        st.subheader("💡 Suggested Home Activity")

        hardest_name, hardest_stats = max(
            achievements.items(),
            key=lambda kv: kv[1].get("frustration_triggers", 0),
        )
        friction = hardest_stats.get("frustration_triggers", 0)
        activity = CONCEPT_ACTIVITIES.get(
            hardest_name,
            f"Revisit '{hardest_name}' together using everyday objects to build intuition!",
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

    # ── Review Queue ──────────────────────────────────────────────────────────
    pending = st.session_state.pending_levels
    if pending:
        st.subheader(f"🗂️ Concept Review Queue ({len(pending)} pending)")
        st.caption("Review each AI-generated concept before adding it to the game.")

        for i, level in enumerate(pending):
            with st.expander(f"📖 {level.get('concept_name', 'Untitled Concept')}", expanded=True):
                st.markdown(f"**Pip's Opening Puzzle:**\n> {level.get('story_intro', '—')}")
                st.markdown(f"**Boss Fight Scenario:**\n> {level.get('verification_scenario', '—')}")

                btn_col1, btn_col2, _ = st.columns([1, 1, 4])
                with btn_col1:
                    if st.button("✅ Add to Game", key=f"add_{i}", use_container_width=True):
                        knowledge = load_knowledge()
                        concepts  = knowledge.setdefault("concepts", {})
                        new_key   = make_concept_key(
                            level.get("concept_name", "New_Concept"),
                            list(concepts.keys()),
                        )
                        # Store the full concept data (without our internal concept_name key)
                        entry = {k: v for k, v in level.items() if k != "concept_name"}
                        concepts[new_key] = entry
                        save_knowledge(knowledge)
                        del st.session_state.pending_levels[i]
                        st.success(f"✅ '{level.get('concept_name')}' added to knowledge.json!")
                        st.rerun()
                with btn_col2:
                    if st.button("🗑️ Discard", key=f"discard_{i}", use_container_width=True):
                        del st.session_state.pending_levels[i]
                        st.rerun()

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

            with st.spinner("AI is designing a puzzle from the PDF…"):
                sys_p, usr_p = compile_concept_generation_prompt(extracted_text)
                raw_json     = call_ollama(sys_p, usr_p, json_mode=True)

            try:
                concept = json.loads(raw_json)
            except json.JSONDecodeError:
                st.error(
                    "The AI returned malformed JSON. Try uploading a cleaner PDF "
                    "or a shorter, more focused excerpt.",
                    icon="❌",
                )
                st.stop()

            # Tag with a display name for the Review Queue expander title
            concept["concept_name"] = concept.get("name", "Untitled Concept")
            st.session_state.pending_levels.append(concept)
            st.success(
                f"✅ **'{concept['concept_name']}'** added to the Review Queue above. "
                "Check the preview and click **Add to Game** when you're happy with it."
            )
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

    # ── Load knowledge graph ──────────────────────────────────────────────────
    try:
        with open("knowledge.json", "r", encoding="utf-8") as fh:
            knowledge = json.load(fh)
    except FileNotFoundError:
        st.error("knowledge.json not found. Run `streamlit run app.py` from the project root.")
        st.stop()
    except json.JSONDecodeError as exc:
        st.error(f"knowledge.json is malformed: {exc}")
        st.stop()

    concepts      = knowledge.get("concepts", {})
    concept_keys  = list(concepts.keys())
    concept_names = [concepts[k]["name"] for k in concept_keys]

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
            for concept_name in achievements:
                st.markdown(f"⭐ **{concept_name}**")
            st.divider()

        # Game controls are only meaningful in Play mode
        if nav_view == "🎮 Play (Kid Mode)":
            selected_idx = st.selectbox(
                "Choose a concept:",
                range(len(concept_names)),
                format_func=lambda i: concept_names[i],
            )

            if st.button("▶ Start / Reset Puzzle", use_container_width=True, type="primary"):
                concept_data = concepts[concept_keys[selected_idx]].copy()

                # Scenario Polymorphism: pick a random variant if available
                if "variants" in concept_data:
                    variant = random.choice(concept_data["variants"])
                    concept_data["story_intro"]               = variant["story_intro"]
                    concept_data["verification_scenario"]     = variant["verification_scenario"]
                    concept_data["verification_ground_truth"] = variant["verification_ground_truth"]

                start_session(concept_data)
                st.rerun()

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

        with st.expander("🔧 Developer Settings"):
            debug_mode = st.checkbox("🐛 Enable Debug Mode", value=False, key="debug_mode")

    # ════════════════════════════════════════════════════════════════════════
    # MAIN AREA — conditional on navigation selection
    # ════════════════════════════════════════════════════════════════════════

    if nav_view == "📊 Dashboard (Parent Mode)":
        render_dashboard()
        st.stop()

    # ── Play (Kid Mode) ───────────────────────────────────────────────────────
    st.title("GemmaGenius")
    st.caption(
        "Explain the concept to Pip — she learns by asking questions, not by being told."
    )

    if not st.session_state.game_started:
        st.info("👈 Choose a concept in the sidebar and click **Start / Reset Puzzle** to begin.")
        st.stop()

    # ── Render full chat history ──────────────────────────────────────────────
    render_chat_history()

    # ── Win state: show celebration and block further input ───────────────────
    if st.session_state.session_won:
        st.balloons()
        st.success("⭐ TRUE MASTERY ACHIEVED! You explained the concept AND caught Pip's mistake!")
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
                    "content": "⚙️ Pip resets the puzzle... Let's try again!",
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

    # ── Chat input ────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Explain it to Pip..."):

        # 1. Append and immediately display the user's message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        concept_data = st.session_state.concept_data

        # Track each student submission during the Boss Fight
        if st.session_state.current_phase == 2:
            st.session_state.boss_fight_attempts += 1

        with st.spinner("Pip is thinking..."):

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

                # Persist the achievement to the student profile
                concept_name = concept_data.get("name", "Unknown Concept")
                st.session_state.profile["achievements"][concept_name] = {
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
                    pip_response = "[System] Pip could not respond. Please try again."

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
                    pip_response = "[System] Pip could not respond. Please try again."

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
