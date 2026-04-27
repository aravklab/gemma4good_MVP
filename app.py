"""
GemmaGenius — Streamlit Web UI
================================
Web UI migration of main.py. All local Ollama API calls are preserved intact.
main.py is NOT modified — this is a brand-new file.

Run with:
    streamlit run app.py
"""

import json
import random
import requests
import streamlit as st


# ---------------------------------------------------------------------------
# CONFIGURATION  (mirrors main.py exactly)
# ---------------------------------------------------------------------------

OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "gemma4:e4b"
TIMEOUT_SECS = 120


# ---------------------------------------------------------------------------
# A. API UTILITY
# ---------------------------------------------------------------------------

def call_ollama(
    system_prompt: str,
    user_prompt:   str,
    json_mode:     bool = False,
) -> str:
    """
    POST to the local Ollama /api/generate endpoint (non-streaming).

    Streaming is omitted here because Streamlit manages its own render loop;
    the spinner replaces the CLI's real-time token output.
    All other parameters and error handling are identical to main.py.
    """
    payload = {
        "model":  MODEL_NAME,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECS)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    except requests.exceptions.Timeout:
        return "[System] Ollama timed out. Is `ollama serve` running?"
    except requests.exceptions.ConnectionError:
        return (
            f"[System] Could not connect to Ollama at {OLLAMA_URL}. "
            "Start it with: `ollama serve`"
        )
    except requests.exceptions.RequestException as exc:
        return f"[System] Unexpected API error: {exc}"


# ---------------------------------------------------------------------------
# B. PROMPT COMPILERS  (identical to main.py — no changes)
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
    """
    Build Pip's system prompt, injecting a dynamic state directive each turn.
    Identical to main.py.
    """
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
# C. ROUTING LOGIC  (mirrors the if/elif blocks inside run_session in main.py)
# ---------------------------------------------------------------------------

def build_state_directive(
    classification:       str,
    concept_data:         dict,
    current_phase:        int,
    frustration_counter:  int,
    clarification_counter: int,
) -> tuple[str, int, int]:
    """
    Map the Evaluator's classification to Pip's state_directive.

    Returns
    -------
    (state_directive, new_frustration_counter, new_clarification_counter)

    Counter rules (exact mirror of main.py):
    - frustration_counter   : increments on every "miss" in Phase 1 only
    - clarification_counter : increments on "question"; resets on any other state
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
            directive = (
                "The user is struggling. React with a confused face and make a "
                "binary-choice guess about what the answer might be, but deliberately "
                "guess the WRONG option so the user feels smart when they correct you."
            )

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

    # Reset clarification counter on any non-question turn (mirrors main.py)
    if classification != "question":
        new_clarification = 0

    return directive, new_frustration, new_clarification


# ---------------------------------------------------------------------------
# D. SESSION STATE INITIALISATION
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialise all st.session_state keys on first load only."""
    defaults = {
        "messages":              [],   # chat history: list of {role, content} dicts
        "current_phase":         1,    # 1 = Elicitation, 2 = Boss Fight
        "frustration_counter":   0,
        "clarification_counter": 0,
        "concept_data":          None,
        "pips_last_question":    "",   # clean text fed to the Evaluator each turn
        "awaiting_retry":        False, # True after give_up — shows Yes/No buttons
        "session_ended":         False,
        "session_won":           False,
        "game_started":          False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def start_session(concept_data: dict) -> None:
    """
    Reset all state and seed Pip's opening line.
    No Ollama call needed for Turn 0 — identical to main.py's design.
    """
    st.session_state.messages              = []
    st.session_state.current_phase         = 1
    st.session_state.frustration_counter   = 0
    st.session_state.clarification_counter = 0
    st.session_state.concept_data          = concept_data
    st.session_state.pips_last_question    = concept_data["story_intro"]
    st.session_state.awaiting_retry        = False
    st.session_state.session_ended         = False
    st.session_state.session_won           = False
    st.session_state.game_started          = True

    # Turn 0: Pip's opening line comes straight from the knowledge graph
    st.session_state.messages.append({
        "role":    "assistant",
        "content": concept_data["story_intro"],
    })


# ---------------------------------------------------------------------------
# E. CHAT HISTORY RENDERER
# ---------------------------------------------------------------------------

def render_chat_history() -> None:
    """
    Render all messages in st.session_state.messages.
    Role values:
      "assistant" → Pip  (avatar 👧🏼)
      "user"      → student (avatar 👤)
      "system"    → phase/state banners rendered as styled st.info boxes
    """
    for message in st.session_state.messages:
        if message["role"] == "system":
            st.info(message["content"], icon="⚔️")
        else:
            avatar = "👧🏼" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])


# ---------------------------------------------------------------------------
# F. MAIN APP
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
        st.markdown("*Teach Pip. Master the concept.*")
        st.divider()

        selected_idx = st.selectbox(
            "Choose a concept:",
            range(len(concept_names)),
            format_func=lambda i: concept_names[i],
        )

        if st.button("▶ Start / Reset Puzzle", use_container_width=True, type="primary"):
            concept_data = concepts[concept_keys[selected_idx]].copy()

            # Scenario Polymorphism: pick a random variant if available —
            # mirrors the identical block in main.py's menu loop exactly.
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

    # ── Main area header ──────────────────────────────────────────────────────
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
    # Mirrors the Python interceptor in main.py's give_up block.
    if st.session_state.awaiting_retry:
        st.markdown("---")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Yes, let's try again!", use_container_width=True):
                st.session_state.awaiting_retry        = False
                st.session_state.frustration_counter   = 0
                st.session_state.clarification_counter = 0

                concept_data = st.session_state.concept_data
                # Re-seed Pip's opening line for the current phase
                if st.session_state.current_phase == 1:
                    opening = concept_data["story_intro"]
                else:
                    opening = concept_data["verification_scenario"]

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
        st.stop()  # Do not show chat_input while awaiting retry decision

    # ── Chat input ────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Explain it to Pip..."):

        # 1. Append and immediately display the user's message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        concept_data = st.session_state.concept_data

        with st.spinner("Pip is thinking..."):

            # 2. Call The Evaluator (Phase-aware, no streaming)
            eval_sys, eval_usr = compile_evaluator_prompt(
                concept_data,
                latest_user_input  = user_input,
                pips_last_question = st.session_state.pips_last_question,
                phase              = st.session_state.current_phase,
            )
            raw_verdict = call_ollama(eval_sys, eval_usr, json_mode=True)

            try:
                classification = (
                    json.loads(raw_verdict)
                    .get("classification", "miss")
                    .lower()
                    .strip()
                )
            except (json.JSONDecodeError, AttributeError):
                classification = "miss"

            # ── Phase 2 Mastery → TRUE WIN STATE ─────────────────────────────
            if st.session_state.current_phase == 2 and classification == "mastery":
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
            # Pip delivers both the Phase 1 validation AND the Boss Fight
            # scenario in one LLM call, avoiding double-printing and context
            # misalignment from injecting raw JSON into the chat history.
            elif st.session_state.current_phase == 1 and classification == "mastery":
                # Update State
                st.session_state.current_phase         = 2
                st.session_state.frustration_counter   = 0
                st.session_state.clarification_counter = 0

                # Inject the Boss Fight banner into chat history so it
                # persists across rerenders and renders via render_chat_history()
                st.session_state.messages.append({
                    "role":    "system",
                    "content": "Phase 1 complete! Boss Fight starting...",
                })

                # Get the Boss Fight Scenario
                boss_scenario = concept_data.get("verification_scenario", "my new scenario")

                # Force Pip to deliver the Boss Fight Scenario directly as her
                # reaction to winning Phase 1
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
                )
                st.session_state.frustration_counter   = new_f
                st.session_state.clarification_counter = new_c

                # 5. Call the Response Generator (Pip)
                pip_system   = compile_pip_prompt(concept_data, directive)
                pip_response = call_ollama(pip_system, user_prompt=user_input, json_mode=False)

                if not pip_response:
                    pip_response = "[System] Pip could not respond. Please try again."

                st.session_state.messages.append({"role": "assistant", "content": pip_response})
                st.session_state.pips_last_question = pip_response

                # If give_up, mark awaiting_retry — next render shows Yes/No buttons
                if classification == "give_up":
                    st.session_state.awaiting_retry = True

                st.rerun()


if __name__ == "__main__":
    main()
