"""
GemmaGenius - Layer 1: Core Feynman Loop
=========================================
Two logical units, one local model (Ollama):
  - Pip (The Actor)      : 8-year-old persona with an Information Firewall.
  - The Evaluator        : Silent background judge; emits strict JSON only.

State is managed with plain Python variables — no agent frameworks.
"""

import json
import random
import sys
import requests

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL_NAME    = "gemma4:e4b"  # swap to any locally available Ollama model
TIMEOUT_SECS  = 120           # per-request wall-clock limit


# ---------------------------------------------------------------------------
# A. API UTILITY
# ---------------------------------------------------------------------------

def call_ollama(
    system_prompt: str,
    user_prompt:   str,
    json_mode:     bool = False,
    stream:        bool = False,
) -> str:
    """
    POST to the local Ollama /api/generate endpoint.

    Parameters
    ----------
    system_prompt : Persona / instruction block for the model.
    user_prompt   : The human-turn text sent this request.
    json_mode     : Enables Ollama's tokeniser-level JSON enforcement.
    stream        : When True, tokens are printed to stdout in real-time
                    and the accumulated string is returned at the end.

    Returns
    -------
    The full response text, or "" on any error.
    """

    payload = {
        "model":  MODEL_NAME,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": stream,
    }

    if json_mode:
        # Ollama's "format" key constrains sampling to valid JSON tokens —
        # far more reliable than just asking politely in the prompt.
        payload["format"] = "json"

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=TIMEOUT_SECS,
            stream=stream,        # keep the HTTP connection open for chunks
        )
        response.raise_for_status()

        # ── Streaming path: print each token as it arrives ──────────────────
        if stream:
            accumulated = []
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                accumulated.append(token)
                if chunk.get("done", False):
                    break
            print()   # newline after Pip finishes speaking
            return "".join(accumulated).strip()

        # ── Non-streaming path: return full payload at once ─────────────────
        return response.json().get("response", "").strip()

    except requests.exceptions.Timeout:
        print("\n[System] Ollama timed out. Is `ollama serve` running?")
        return ""
    except requests.exceptions.ConnectionError:
        print(f"\n[System] Could not connect to Ollama at {OLLAMA_URL}.")
        print("[System] Start it with: `ollama serve`")
        return ""
    except requests.exceptions.RequestException as exc:
        print(f"\n[System] Unexpected API error: {exc}")
        return ""


# ---------------------------------------------------------------------------
# B. PROMPT COMPILERS
# ---------------------------------------------------------------------------

def compile_evaluator_prompt(
    concept_data:       dict,
    latest_user_input:  str,
    pips_last_question: str,
    phase:              int = 1,
) -> tuple[str, str]:
    """
    Build the (system, user) prompt pair for The Evaluator.

    The Evaluator sees ONLY:
      - The ground truth it is marking against (phase-dependent).
      - Pip's most recent question (for context).
      - The student's latest reply.

    It MUST return exactly one JSON key: "classification".
    Valid values: "mastery" | "partial_hit" | "miss" | "question" | "off_topic" | "give_up"

    phase=1  → grade against evaluator_ground_truth  (Elicitation)
    phase=2  → grade against verification_ground_truth (Boss Fight)
    """

    # Select the correct rubric for the current phase.
    if phase == 2:
        ground_truth = concept_data["verification_ground_truth"]
    else:
        ground_truth = concept_data["evaluator_ground_truth"]

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

    The persona and firewall are fully generic — all concept-specific context
    comes from the knowledge graph via story_intro and state_directive.
    """

    system_prompt = f"""You are Pip, an enthusiastic 8-year-old trying to solve a puzzle.
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

    return system_prompt


# ---------------------------------------------------------------------------
# C. THE CORE CLI ENGINE
# ---------------------------------------------------------------------------

def run_session(concept_data: dict) -> None:
    """
    Run one full Feynman-Technique teaching session for a single concept.

    State machine per turn
    ──────────────────────
    Evaluator classifies input → routing logic selects state_directive →
    Pip responds with directive injected → loop continues until mastery.

    Counters
    ────────
    frustration_counter   : increments on every "miss"
    clarification_counter : increments on "question"; resets on any other state
    """

    # ── State variables ──────────────────────────────────────────────────────
    frustration_counter   = 0
    clarification_counter = 0
    current_phase         = 1   # 1 = Elicitation, 2 = Verification Boss Fight
    pips_last_question    = concept_data["story_intro"]   # seeded with Turn 0

    # ── Turn 0: Print the story intro WITHOUT calling Ollama ────────────────
    print("\n" + "=" * 62)
    print(f"  GemmaGenius  ·  Concept: {concept_data['name']}")
    print("=" * 62)
    print("\n[Tip: Explain the concept back to Pip in your own words.]\n")
    print(f"Pip: {concept_data['story_intro']}\n")

    # ── Main teaching loop ───────────────────────────────────────────────────
    while True:

        # 1. Capture user input ───────────────────────────────────────────────
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[Session ended by user]")
            sys.exit(0)

        if not user_input:
            continue

        # 2. Run The Evaluator ────────────────────────────────────────────────
        print(f"\n[Evaluator] Phase {current_phase} — Judging ...", end=" ", flush=True)

        eval_system, eval_user = compile_evaluator_prompt(
            concept_data,
            latest_user_input  = user_input,
            pips_last_question = pips_last_question,
            phase              = current_phase,
        )
        raw_verdict = call_ollama(eval_system, eval_user, json_mode=True, stream=False)

        # ── Parse JSON verdict ───────────────────────────────────────────────
        try:
            verdict        = json.loads(raw_verdict)
            classification = verdict.get("classification", "miss").lower().strip()
        except json.JSONDecodeError:
            # Malformed output — treat as a miss so the session continues.
            print(f"(parse error: {raw_verdict!r}) → defaulting to 'miss'")
            classification = "miss"

        print(f"verdict = {classification.upper()}")

        # 3. Routing Logic ────────────────────────────────────────────────────

        # ════════════════════════════════════════════════════════════════════
        # PHASE 1 — ELICITATION
        # Goal: get the student to describe what a right angle looks like.
        # ════════════════════════════════════════════════════════════════════
        if current_phase == 1:

            if classification == "mastery":
                # ── Phase 1 cleared → instant transition, zero API latency ──
                print(f"\nPip: {concept_data['story_bridge']}")
                print("\n[⚙️  Pip is resetting her blocks... Building the Boss Fight Stage...]\n")

                # Present the verification scenario directly from JSON — no Ollama call.
                print(f"Pip: {concept_data['verification_scenario']}\n")

                # Seed pips_last_question so the Evaluator has context next turn.
                pips_last_question = concept_data["verification_scenario"]
                current_phase = 2
                print("[State] Phase transition → PHASE 2 (Verification Boss Fight)")
                continue  # Wait for the user's response before calling Ollama

            elif classification == "partial_hit":
                state_directive = (
                    "The user is getting close. Acknowledge their specific analogy, but "
                    "YOU MUST IMMEDIATELY connect it back to the physical building blocks. "
                    "Ask them to describe the straight lines or edges of whatever object "
                    "they just mentioned to force them to explain the geometry."
                )

            elif classification == "miss":
                frustration_counter += 1
                print(f"[State] frustration_counter = {frustration_counter}")
                state_directive = (
                    "The user is struggling. React with a confused face and make a "
                    "binary-choice guess, but deliberately guess the WRONG option "
                    "(e.g. 'Is it like a wiggly line, or a really sharp spiky shape?') "
                    "so the user feels smart when they correct you."
                )

            elif classification == "off_topic":
                state_directive = (
                    "The user is distracted. Acknowledge what they said like a kid would, "
                    "but immediately pivot back to your current puzzle. Remind them you "
                    "really need their help to figure this out."
                )

            elif classification == "question":
                if clarification_counter < 2:
                    clarification_counter += 1
                    print(f"[State] clarification_counter = {clarification_counter}")
                    state_directive = (
                        "The user is confused about something you said. Briefly clarify "
                        "using completely different, simpler words. Do NOT give the answer."
                    )
                else:
                    state_directive = (
                        "The user keeps asking questions and you are both going in circles. "
                        "Tell them you are too confused too, and suggest that maybe you both "
                        "need to look at an actual book together to figure it out."
                    )

            elif classification == "give_up":
                # ── Recovery Loop: reveal the answer, offer a retry ──────────
                # Phase-Aware Generic Escape Hatch
                if current_phase == 1:
                    ans = concept_data.get("evaluator_ground_truth", "the correct answer")
                    state_directive = (
                        f"The user is frustrated. Tell them the answer clearly based on "
                        f"this rule: {ans}. Then cheerfully ask if they want to try "
                        f"explaining it so you can finish your task."
                    )
                else:
                    ans = concept_data.get("verification_ground_truth", "why your scenario was wrong")
                    state_directive = (
                        f"The user is frustrated. Tell them exactly why your scenario was "
                        f"wrong based on this rule: {ans}. Then cheerfully ask if they want "
                        f"to try explaining it so you can finish your task."
                    )
                pip_system = compile_pip_prompt(concept_data, state_directive)
                print("\n[Pip is thinking...]")
                print("\nPip: ", end="", flush=True)
                call_ollama(pip_system, user_prompt=user_input, json_mode=False, stream=True)

                # Python interceptor — no AI needed for yes/no routing
                retry_choice = input("\n\nYou: ").strip().lower()

                if retry_choice in ["yes", "y", "sure", "ok", "okay", "yeah", "i guess"]:
                    print("\n[⚙️  Pip is resetting the puzzle... Let's try again!]\n")
                    frustration_counter   = 0
                    clarification_counter = 0
                    # Re-seed Pip's opening line for the current phase
                    if current_phase == 1:
                        print(f"Pip: {concept_data['story_intro']}\n")
                        pips_last_question = concept_data["story_intro"]
                    else:
                        print(f"Pip: {concept_data['verification_scenario']}\n")
                        pips_last_question = concept_data["verification_scenario"]
                    continue  # Jump back to the top of the main loop

                else:
                    print("\n[Session ended gracefully. You did a great job trying!]")
                    return  # Exit run_session cleanly

            else:
                print(f"[State] Unknown classification '{classification}' — defaulting to miss.")
                state_directive = (
                    "The user said something unexpected. Gently bring the conversation "
                    "back to your block-house puzzle."
                )
        # Goal: student must catch that Pip's roof idea is WRONG.
        # ════════════════════════════════════════════════════════════════════
        elif current_phase == 2:

            if classification == "mastery":
                # ── TRUE WIN STATE ───────────────────────────────────────────
                print("\n" + "★" * 62)
                print("  ★  TRUE MASTERY ACHIEVED!                               ★")
                print("  ★  You explained the concept AND caught Pip's mistake!  ★")
                print("★" * 62 + "\n")
                break

            elif classification == "partial_hit":
                state_directive = (
                    "The user caught part of your mistake, but didn't fully explain "
                    "why your roof idea is wrong. Ask them to clarify why an 'L' shape "
                    "wouldn't make a flat roof."
                )

            elif classification == "miss":
                # No frustration penalty in Phase 2 — this is a different challenge.
                state_directive = (
                    "The user didn't catch your mistake. Act genuinely confused about "
                    "your own roof idea and ask if an 'L' shape would really make a "
                    "flat roof, or if it would make something else."
                )

            elif classification == "off_topic":
                state_directive = (
                    "The user is distracted. Acknowledge what they said like a kid would, "
                    "but immediately pivot back to your current puzzle. Remind them you "
                    "really need their help to figure this out."
                )

            elif classification == "question":
                if clarification_counter < 2:
                    clarification_counter += 1
                    print(f"[State] clarification_counter = {clarification_counter}")
                    state_directive = (
                        "The user is confused about your roof question. Rephrase it "
                        "using simpler words. Do NOT reveal whether the roof idea is "
                        "right or wrong."
                    )
                else:
                    state_directive = (
                        "You and the user are both confused. Suggest you draw it out "
                        "on paper together to see what the roof would actually look like."
                    )

            elif classification == "give_up":
                # ── Recovery Loop: reveal the answer, offer a retry ──────────
                # Phase-Aware Generic Escape Hatch
                if current_phase == 1:
                    ans = concept_data.get("evaluator_ground_truth", "the correct answer")
                    state_directive = (
                        f"The user is frustrated. Tell them the answer clearly based on "
                        f"this rule: {ans}. Then cheerfully ask if they want to try "
                        f"explaining it so you can finish your task."
                    )
                else:
                    ans = concept_data.get("verification_ground_truth", "why your scenario was wrong")
                    state_directive = (
                        f"The user is frustrated. Tell them exactly why your scenario was "
                        f"wrong based on this rule: {ans}. Then cheerfully ask if they want "
                        f"to try explaining it so you can finish your task."
                    )
                pip_system = compile_pip_prompt(concept_data, state_directive)
                print("\n[Pip is thinking...]")
                print("\nPip: ", end="", flush=True)
                call_ollama(pip_system, user_prompt=user_input, json_mode=False, stream=True)

                # Python interceptor — no AI needed for yes/no routing
                retry_choice = input("\n\nYou: ").strip().lower()

                if retry_choice in ["yes", "y", "sure", "ok", "okay", "yeah", "i guess"]:
                    print("\n[⚙️  Pip is resetting the puzzle... Let's try again!]\n")
                    frustration_counter   = 0
                    clarification_counter = 0
                    # Re-seed Pip's opening line for the current phase
                    if current_phase == 1:
                        print(f"Pip: {concept_data['story_intro']}\n")
                        pips_last_question = concept_data["story_intro"]
                    else:
                        print(f"Pip: {concept_data['verification_scenario']}\n")
                        pips_last_question = concept_data["verification_scenario"]
                    continue  # Jump back to the top of the main loop

                else:
                    print("\n[Session ended gracefully. You did a great job trying!]")
                    return  # Exit run_session cleanly

            else:
                print(f"[State] Unknown classification '{classification}' — defaulting to miss.")
                state_directive = (
                    "The user said something unexpected. Bring the conversation back "
                    "to whether your flat roof idea is correct."
                )

        # 4. Reset clarification counter on any non-question turn ─────────────
        if classification != "question":
            if clarification_counter > 0:
                print(f"[State] clarification_counter reset (was {clarification_counter})")
            clarification_counter = 0

        # 5. Compile Pip's prompt with the selected directive ─────────────────
        pip_system = compile_pip_prompt(concept_data, state_directive)

        # 6. Stream Pip's response to the terminal ────────────────────────────
        print("\n[Pip is thinking...]")
        print("\nPip: ", end="", flush=True)
        pip_response = call_ollama(
            pip_system,
            user_prompt = user_input,   # Pip "hears" what the student just said
            json_mode   = False,
            stream      = True,
        )

        if not pip_response:
            print("[System] Pip could not respond. Please try again.\n")
            continue

        # Seed pips_last_question for the next evaluator call
        pips_last_question = pip_response
        print()   # visual breathing room before next input prompt


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    """Load the knowledge graph, show the concept menu, and run sessions."""

    print("\n[STAGE 1/2] Loading knowledge.json ...")
    try:
        with open("knowledge.json", "r", encoding="utf-8") as fh:
            knowledge = json.load(fh)
    except FileNotFoundError:
        print("[Error] knowledge.json not found. Run from the project root.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[Error] knowledge.json is malformed: {exc}")
        sys.exit(1)

    concepts = knowledge.get("concepts", {})
    if not concepts:
        print("[Error] No concepts found in knowledge.json.")
        sys.exit(1)

    concept_keys = list(concepts.keys())
    print(f"[STAGE 1/2] OK — {len(concepts)} concept(s) loaded.")

    # ── Outer menu loop ───────────────────────────────────────────────────────
    # After each session ends (mastery or escape hatch), execution returns here
    # and the menu is shown again. Type 'q' to exit the program entirely.
    while True:
        print("\n" + "=" * 62)
        print("  GemmaGenius  ·  Main Menu")
        print("=" * 62)
        print("\nAvailable concepts:\n")
        for i, key in enumerate(concept_keys, start=1):
            print(f"  [{i}] {concepts[key]['name']}")
        print("\n  [q] Quit\n")

        try:
            choice = input("Select a concept: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            sys.exit(0)

        # ── Quit ─────────────────────────────────────────────────────────────
        if choice in ("q", "quit"):
            print("\nGoodbye!")
            sys.exit(0)

        # ── Validate numeric input ────────────────────────────────────────────
        if not choice.isdigit() or not (1 <= int(choice) <= len(concept_keys)):
            print(f"\n[Error] Enter a number between 1 and {len(concept_keys)}, or 'q' to quit.")
            continue

        # ── Load selected concept and start session ───────────────────────────
        selected_key = concept_keys[int(choice) - 1]
        concept_data = concepts[selected_key].copy()  # copy so variants don't mutate the source

        # ── Scenario Polymorphism: pick a random variant if available ─────────
        # Variants override story_intro, verification_scenario, and
        # verification_ground_truth so the same concept feels fresh each run.
        # Backwards-compatible: concepts without "variants" are unchanged.
        if "variants" in concept_data:
            variant = random.choice(concept_data["variants"])
            concept_data["story_intro"]               = variant["story_intro"]
            concept_data["verification_scenario"]     = variant["verification_scenario"]
            concept_data["verification_ground_truth"] = variant["verification_ground_truth"]
            print(f"\n[Variant] Selected scenario: '{variant.get('variant_name', 'unnamed')}'")

        print(f"\n[STAGE 2/2] Starting: '{concept_data['name']}' ({selected_key})")
        print("[STAGE 2/2] No Ollama call needed for Turn 0.\n")

        run_session(concept_data)
        # run_session returns here naturally when:
        #   - Phase 2 mastery break exits the inner while, function falls through
        #   - Escape hatch "no" path calls return directly
        # Either way execution continues to the next outer loop iteration → menu


if __name__ == "__main__":
    main()
