# GemmaGenius

A local, privacy-first educational app built on the **Feynman Technique** — the student explains a concept back to an AI, and the AI teaches by asking questions, not by giving answers.

Everything runs on your machine. No cloud APIs. No data leaves your device.

Two interfaces are available — both use the same local Ollama model and the same `knowledge.json` knowledge graph:

| Interface | File | How to run |
|---|---|---|
| Web UI (Streamlit) | `app.py` | `streamlit run app.py` |
| CLI | `main.py` | `python main.py` |

---

## How It Works

The core idea is that the best way to test understanding is to make the learner *teach*. GemmaGenius puts the student in the role of teacher: a confused 8-year-old named **Pip** asks them to explain a math concept. The student has to find the right words, and Pip reacts based on how close they are.

Two AI units share the same local model but serve completely different roles:

```
                    ┌─────────────────────────────────┐
                    │   WEB UI (sidebar selectbox)     │
                    │   or CLI MAIN MENU               │
                    │  [1] The Book Corner             │
                    │  [2] The Great Ground-Puller     │
                    └──────────────┬──────────────────┘
                                   │ concept selected
                                   ▼
Student types an answer
        │
        ▼
┌─────────────────┐     JSON verdict         ┌──────────────────┐
│   The Evaluator │ ──────────────────────▶  │  Routing Logic   │
│  (silent judge) │  mastery / miss /        │  (plain Python)  │
└─────────────────┘  partial_hit /           └────────┬─────────┘
                     question /                       │ state_directive
                     off_topic /                      ▼
                     give_up              ┌─────────────────────┐
                                          │     Pip (Actor)     │
                                          │  8-year-old         │ ──▶ Web UI chat bubble
                                          │  puzzle-solver      │     or streamed to terminal
                                          └─────────────────────┘
                                                      │
                                          session ends (mastery / exit)
                                                      │
                                                      ▼
                                          back to concept selector
```

---

## Session Flow (Two-Phase Structure)

A complete session has two acts, identical in both the Web UI and CLI:

```
  CONCEPT SELECTOR
  ────────────────
  [1] The Book Corner
  [2] The Great Ground-Puller
        │ user picks a concept
        ▼
PHASE 1 — ELICITATION
  Pip presents story_intro (verbatim from JSON, no Ollama call)
  → Student must explain the concept correctly (mastery)
        │
        ▼  (seamless transition — single Ollama call)
  Pip enthusiastically validates the student, then IMMEDIATELY
  presents a DELIBERATE MISTAKE from the verification_scenario.
        │
        ▼
PHASE 2 — VERIFICATION BOSS FIGHT
  Pip presents verification_scenario embedded in her reaction
  → Student must catch and correct Pip's wrong application
        │
        ▼  (Phase 2 mastery)
  ★ TRUE MASTERY ACHIEVED ★  →  back to concept selector

  ─── At any point ───
  Student types "I give up" / "just tell me" / expresses frustration
        │
        ▼  (give_up escape hatch — one Ollama call with OVERRIDE FIREWALL)
  Pip reveals the answer (pulled from JSON ground truth) and asks:
  "Now that you know the secret, do you want to try explaining it
   to me so we can finish?"
        │
        ├── yes / sure / ok / yeah  →  counters reset, puzzle replays from
        │                              current phase opening line, loop continues
        └── anything else           →  "Session ended gracefully."  →  back to selector
```

---

## Project Structure

```
Gemma4good/
├── app.py                        # Web UI — Streamlit interface (feature/ui-improvements branch)
├── main.py                       # CLI engine — original interface, untouched
├── knowledge.json                # Concept knowledge graph (shared by both interfaces)
├── Master Ingestion Prompt.txt   # LLM prompt for generating new concepts
├── Math_chapter1.md              # Example source curriculum (Grade 4 Geometry)
├── ARCHITECTURE.md               # Design rules and constraints
└── README.md                     # This file
```

---

## What Has Been Built

### `knowledge.json` — The Knowledge Graph

A structured JSON file that stores teachable concepts. Each concept contains:

| Field | Purpose |
|---|---|
| `name` | Display name shown in the session header and main menu |
| `story_bridge` | Pip's Phase 1 win reaction — printed on elicitation mastery |
| `secret_fact` | The answer Pip secretly knows but must never say outright |
| `goal` | Plain-English description of what Pip is nudging the student toward |
| `evaluator_ground_truth` | Rubric the Evaluator grades against in Phase 1 |
| `story_intro` | *(root-level)* Pip's opening line if no `variants` array is present |
| `verification_scenario` | *(root-level)* Pip's deliberate-mistake question if no `variants` array is present |
| `verification_ground_truth` | *(root-level)* Phase 2 rubric if no `variants` array is present |
| `variants` | *(optional)* Array of scenario objects — enables **Scenario Polymorphism** |

**Scenario Polymorphism (`variants` array)**

When a concept has a `"variants"` key, the engine picks one at random each session using `random.choice()` and injects that variant's `story_intro`, `verification_scenario`, and `verification_ground_truth` into the working copy of `concept_data`. The root-level fields (`name`, `story_bridge`, `evaluator_ground_truth`, etc.) are shared across all variants.

Each variant object contains:

| Field | Purpose |
|---|---|
| `story_intro` | Pip's opening line for this specific scenario |
| `verification_scenario` | Pip's deliberate-mistake question for this scenario |
| `verification_ground_truth` | Phase 2 rubric specific to this scenario's mistake |
| `variant_name` | *(optional)* Label shown in the `[Variant]` log line |

Backwards-compatible: concepts without `"variants"` use their root-level fields unchanged.

**Current concepts:** 2

| Key | Name | Variants |
|---|---|---|
| `C1_Right_Angle` | The Book Corner | — (single fixed scenario) |
| `C3_New_Concept` | The Earth's Invisible Tug | 3 (trampoline, ceiling stickers, block bridge) |

---

### `main.py` — The CLI Engine (original)

#### A. `call_ollama(system_prompt, user_prompt, json_mode, stream)`

A single function that handles all communication with the local Ollama API.

- **`json_mode=True`** — sets Ollama's `"format": "json"` field, which constrains token sampling to valid JSON at the model level. Used for the Evaluator.
- **`stream=True`** — keeps the HTTP connection open and prints tokens to the terminal in real time as they arrive. Used for Pip, so her replies feel live.
- Handles `Timeout`, `ConnectionError`, and generic request errors with clear actionable messages.
#### B. `compile_evaluator_prompt(concept_data, latest_user_input, pips_last_question, phase)`

Builds the silent judge's prompt. The `phase` parameter selects the correct rubric:
- `phase=1` → grades against `evaluator_ground_truth`
- `phase=2` → grades against `verification_ground_truth`

The Evaluator sees only three things: the active rubric, Pip's last question, and the student's latest message. It outputs **only** `{"classification": "<state>"}`.

The six classification states:

| State | Meaning |
|---|---|
| `mastery` | Student clearly explained the core mechanism required by the ground truth (childlike language is fine) |
| `partial_hit` | Student mentions relevant ideas or vocabulary but the explanation is incomplete, vague, or missing the core logical mechanism |
| `miss` | Student is wrong, confused, guessing blindly, or merely parroting vocabulary without explaining how it works |
| `question` | Student asked a clarifying question about the concept or puzzle |
| `off_topic` | Joke, nonsense, or completely unrelated topic |
| `give_up` | Student explicitly surrenders ("I don't know", "just tell me", expresses deep frustration). A wrong answer is still `miss` — this only fires on a clear, explicit surrender. Triggers the **Recovery Loop** (see below) |

#### C. `compile_pip_prompt(concept_data, state_directive)`

Builds Pip's system prompt each turn with a fresh `state_directive` injected at the end.

**Generic persona:** Pip is now a subject-neutral "8-year-old trying to solve a puzzle." All concept-specific context comes from `story_intro` (Turn 0) and the `state_directive` — no hardcoded subject references exist in Python.

**Conditional Information Firewall:** The base rule is:
> *"Unless your specific State Directive explicitly tells you to give the answer, you must NEVER give away the exact answer or vocabulary word."*

This means the firewall is **on by default** but can be overridden by directives that open with `OVERRIDE FIREWALL:`. Normal turns keep the firewall active; the `give_up` escape hatch overrides it so Pip can reveal the answer without fighting the base persona.

#### D. `run_session(concept_data)` — The Two-Phase State Machine

The `while True` loop with three state variables:

| Variable | Purpose |
|---|---|
| `current_phase` | `1` = Elicitation, `2` = Verification Boss Fight |
| `frustration_counter` | Increments on every Phase 1 `miss` (reserved for future hint ladder) |
| `clarification_counter` | Increments on `question`; resets on any non-`question` state |

#### E. `main()` — Outer Menu Loop + Scenario Polymorphism

Loads `knowledge.json` once, then runs an outer `while True` menu loop:

1. Prints a numbered list of all concept names, built dynamically from the JSON keys.
2. User selects a number or types `q` to quit.
3. **Scenario Polymorphism:** calls `concept_data = concepts[key].copy()`, then checks for a `"variants"` key. If present, `random.choice()` picks a variant and its three fields (`story_intro`, `verification_scenario`, `verification_ground_truth`) are injected into the working copy. If absent, the root fields are used unchanged.
4. Calls `run_session(concept_data)`. When it returns (mastery or escape hatch exit), the outer loop shows the menu again.

**Phase 1 routing:**

| Evaluator verdict | Counter effect | Pip's behaviour |
|---|---|---|
| `mastery` | — | Prints `story_bridge`, prints `verification_scenario` verbatim from JSON (zero Ollama latency), sets `current_phase = 2`, `continue`s |
| `partial_hit` | none | Enthusiastically validates what the student got right, then asks a guiding question to help them find the missing piece — never gives the answer |
| `miss` | `frustration++` | Makes a deliberately wrong binary guess so the student feels smart correcting her |
| `off_topic` | none | Acknowledges like a kid, then re-states the exact `story_intro` puzzle verbatim to prevent context drift |
| `question` (1st or 2nd) | `clarification++` | Answers briefly in character, then re-states the exact `story_intro` puzzle verbatim so context is never lost |
| `question` (3rd+) | none | "Maybe we should look at a book together" |
| `give_up` | — | Reveals the full answer warmly using the `secret_fact` analogy, then asks if the student wants to retry. **Yes** → resets counters, re-prints the phase opening line, `continue`s. **No** → `return`s |

**Phase 2 routing:**

| Evaluator verdict | Counter effect | Pip's behaviour |
|---|---|---|
| `mastery` | — | Prints `★ TRUE MASTERY ACHIEVED ★` banner and breaks |
| `partial_hit` | none | Enthusiastically validates what the student got right, then asks a guiding question to find the missing piece |
| `miss` | none (no penalty) | Acts confused, asks the student to explain what would actually happen if her scenario idea were used |
| `off_topic` | none | Acknowledges like a kid, then re-states the exact `verification_scenario` verbatim to prevent context drift |
| `question` (1st or 2nd) | `clarification++` | Answers briefly in character, then re-states the exact `verification_scenario` verbatim so context is never lost |
| `question` (3rd+) | none | Suggests drawing it out on paper together |
| `give_up` | — | Reveals the full answer warmly using `verification_ground_truth`, then asks if the student wants to retry. **Yes** → resets counters, re-prints the verification scenario, `continue`s. **No** → `return`s |

**Recovery Loop detail (`give_up` path):**

1. One Ollama call — directive opens with `OVERRIDE FIREWALL:` to bypass the base persona's answer prohibition.
   - **Phase 1:** Pip explains the answer using the `secret_fact` analogy in a full, helpful paragraph.
   - **Phase 2:** Pip explains exactly why her scenario was wrong using `verification_ground_truth` in a full, helpful paragraph.
   - Both phases end with the verbatim question *"Now that we know the secret, do you want to try explaining it to me so we can finish?"*
2. A plain Python `input()` intercepts the reply — no AI needed for yes/no routing.
3. **Yes** (`yes / y / sure / ok / okay / yeah / i guess`) — resets `frustration_counter` and `clarification_counter` to 0, re-seeds `pips_last_question` with the correct phase opening line, prints it, and `continue`s back to the top of the loop.
4. **Anything else** — prints `[Session ended gracefully. You did a great job trying!]`, `return`s from `run_session`, and the outer menu loop shows the concept selector again.

**Context Re-Injection:** When the Evaluator returns `question` or `off_topic`, Pip's `state_directive` always includes the exact verbatim text of the current phase's puzzle (`story_intro` for Phase 1, `verification_scenario` for Phase 2). This prevents the LLM from drifting to a hallucinated version of the puzzle after several turns.

**UX indicator:** `[Pip is thinking...]` is printed before every streaming Ollama call so the student knows a response is coming rather than seeing a silent pause.

**Turn 0** prints `story_intro` directly — no Ollama call, so the app starts instantly.

---

### `app.py` — The Streamlit Web UI

A full migration of `main.py`'s logic into a Streamlit web application. `main.py` is **not modified** — `app.py` is a separate file that re-implements the same two-phase state machine using Streamlit's paradigms. Both files share the same prompt compilers and Ollama calls.

#### Paradigm shift: session state replaces the `while` loop

Streamlit reruns the entire script top-to-bottom on every user interaction. All variables that `main.py` holds in the `while True` loop are instead stored in `st.session_state`:

| `main.py` variable | `st.session_state` key | Default |
|---|---|---|
| `current_phase` | `current_phase` | `1` |
| `frustration_counter` | `frustration_counter` | `0` |
| `clarification_counter` | `clarification_counter` | `0` |
| `pips_last_question` | `pips_last_question` | `""` |
| `concept_data` (local) | `concept_data` | `None` |
| Chat output (terminal) | `messages` (list of dicts) | `[]` |
| — | `awaiting_retry` | `False` |
| — | `session_ended` | `False` |
| — | `session_won` | `False` |
| — | `game_started` | `False` |

#### Interface mapping

| `main.py` (CLI) | `app.py` (Web UI) |
|---|---|
| Numbered concept menu | `st.sidebar` selectbox |
| `input("You: ")` | `st.chat_input("Explain it to Pip...")` |
| `print("Pip: ...")` with streaming | `st.chat_message("assistant", avatar="👧🏼")` |
| `[Pip is thinking...]` print | `with st.spinner("Pip is thinking..."):` |
| `print("★" * 62)` win banner | `st.balloons()` + `st.success()` |
| `input()` yes/no after give_up | **Yes / No buttons** (chat input hidden via `st.stop()`) |
| Phase transition `print()` notices | `"system"` role messages rendered as `st.info()` boxes |

#### Phase 1 → Phase 2 transition (seamless single turn)

In `main.py`, Phase 1 mastery prints `story_bridge` and `verification_scenario` directly from JSON with zero Ollama latency. In `app.py` this was redesigned to avoid double-printing and context misalignment: on Phase 1 mastery, Pip is given a single `OVERRIDE FIREWALL` directive that instructs her to enthusiastically validate the student **and** immediately present the `verification_scenario` in her own voice. This produces a more natural, connected handoff in the chat interface.

#### Concept-agnostic routing directives

All `state_directive` strings in `app.py` are generic — no hardcoded references to specific concepts, shapes, or physical objects. Pip's character and the active `story_intro` / `verification_scenario` text (injected dynamically from `knowledge.json`) provide all the context she needs.

| Classification | Phase 1 directive summary | Phase 2 directive summary |
|---|---|---|
| `partial_hit` | Validate what they got right, ask guiding question, no answer | Validate partial debug logic, ask follow-up to fully debunk the scenario |
| `miss` | Confused face, deliberate wrong binary guess so student feels smart correcting | Act genuinely confused, ask a follow-up that nudges them toward the flaw |
| `off_topic` | Acknowledge like a kid, re-state exact `story_intro` verbatim | Acknowledge like a kid, re-state exact `verification_scenario` verbatim |
| `question` (1–2) | Answer briefly in character, re-state exact `story_intro` | Answer briefly in character, re-state exact `verification_scenario` |
| `question` (3+) | "Maybe we should look at a book together" | "Maybe we should draw it out on paper" |
| `give_up` | OVERRIDE FIREWALL: reveal answer using `secret_fact` analogy, ask Yes/No retry | OVERRIDE FIREWALL: reveal answer using `verification_ground_truth`, ask Yes/No retry |
| `mastery` (Phase 1) | OVERRIDE FIREWALL: validate the student, immediately present `verification_scenario` as Pip's new scenario | — |
| `mastery` (Phase 2) | — | `st.balloons()` + TRUE MASTERY win state |

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com) installed and running locally
- A model pulled (default: `gemma4:e4b`)
- Python 3.10+
- `requests` and `streamlit` libraries

```powershell
pip install requests streamlit
ollama pull gemma4:e4b
ollama serve
```

### Run — Web UI (recommended)

```powershell
cd "c:\Projects with Agents\Gemma4good"
streamlit run app.py
```

### Run — CLI

```powershell
cd "c:\Projects with Agents\Gemma4good"
python main.py
```

### Swap the model

Edit the `MODEL_NAME` constant at the top of either `app.py` or `main.py`:

```python
MODEL_NAME = "gemma4:e4b"   # change to any model you have pulled locally
```

---

## Example Session

```
[STAGE 1/2] Loading knowledge.json ...
[STAGE 1/2] OK — 2 concept(s) loaded.

==============================================================
  GemmaGenius  ·  Main Menu
==============================================================

Available concepts:

  [1] The Book Corner
  [2] The Earth's Invisible Tug

  [q] Quit

Select a concept: 2

[Variant] Selected scenario: 'unnamed'   ← random variant picked
[STAGE 2/2] Starting: 'The Earth's Invisible Tug' (C3_New_Concept)
[STAGE 2/2] No Ollama call needed for Turn 0.

==============================================================
  GemmaGenius  ·  Concept: The Earth's Invisible Tug
==============================================================

Pip: I'm out on the trampoline trying to do a 'moon jump,' but no matter
     how hard I launch myself, I keep getting snapped back down to the mat.
     Why does the ground keep winning the tug-of-war?

You: there's a force that pulls everything down toward the earth

[Evaluator] Phase 1 — Judging ... verdict = MASTERY

Pip: My big brother says there is a scientific reason why my toys don't
     just stay where I put them in the air. Can you explain what is
     actually happening?

[⚙️  Pip is resetting her blocks... Building the Boss Fight Stage...]

Pip: Okay, so it's a pull! If I put on my super-heavy winter boots, the
     ground-pull will get tired and let go, letting me float. Right?

You: no the pull gets stronger with heavier things, not weaker

[Evaluator] Phase 2 — Judging ... verdict = MASTERY

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
  ★  TRUE MASTERY ACHIEVED!                               ★
  ★  You explained the concept AND caught Pip's mistake!  ★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

==============================================================
  GemmaGenius  ·  Main Menu        ← returns here automatically
==============================================================
```

---

## Architecture Constraints

Defined in `ARCHITECTURE.md` — enforced throughout:

- **No agent frameworks** — no LangChain, LangGraph, AutoGen, or OpenAI SDK.
- **Raw Python + `requests` only** — one dependency for all HTTP calls.
- **Procedural state** — `while` loops and plain variables; no state machines or graphs.
- **Two logical units, one model** — Pip and the Evaluator are prompt roles, not separate processes.
- **Local knowledge graph** — flat JSON file; no vector DB, no embeddings.

---

## Adding a New Concept to `knowledge.json`

### Using the Master Ingestion Prompt

The fastest way to create new content is with `Master Ingestion Prompt.txt`. Paste that prompt into any capable LLM (Claude, GPT-4, Gemini, etc.) and fill in three placeholders:

```
SUBJECT:     [e.g. Grade 4 Maths]
CONCEPT:     [e.g. Fractions]
CONCEPT KEY: [e.g. C4_Fractions]
```

The model will output a ready-to-paste JSON block with:
- All required root fields (`name`, `secret_fact`, `goal`, `evaluator_ground_truth`, `story_bridge`)
- 3 replayable `variants`, each with a unique `story_intro`, `verification_scenario`, and `verification_ground_truth`
- The `evaluator_ground_truth` anti-repetition rule pre-included

The `Math_chapter1.md` cheat sheet is an example source document — feed it to the LLM alongside the prompt to generate concepts grounded in specific curriculum content.

### Manual Authoring Rules

Each new concept must be a **direct key** inside the `"concepts"` object — do not wrap it in an extra `{ }`:

```json
// CORRECT
"concepts": {
  "C1_Right_Angle": { ... },
  "C4_My_New_Concept": { ... }
}

// WRONG — extra wrapping object causes a parse error
"concepts": {
  "C1_Right_Angle": { ... },
  {
    "C4_My_New_Concept": { ... }
  }
}
```

After editing, always validate:

```powershell
python -c "import json; d=json.load(open('knowledge.json')); print(list(d['concepts'].keys()))"
```

---

## What's Next (Planned Layers)

- **Layer 2 (done):** Streamlit Web UI — `app.py` replaces the CLI with a chat interface, sidebar concept selector, and `st.session_state` state machine. Branch: `feature/ui-improvements`.
- **Layer 3:** Frustration escalation — use `frustration_counter` to trigger Pip's "hint ladder" (progressively stronger hints without giving the answer).
- **Layer 4:** Session scoring — report mastery rate, number of misses, and time-to-mastery at the end of each concept.
- **Layer 5:** Concept authoring CLI — add new concepts to `knowledge.json` interactively without editing the file by hand.
