# GemmaGenius

A local, privacy-first educational CLI app built on the **Feynman Technique** — the student explains a concept back to an AI, and the AI teaches by asking questions, not by giving answers.

Everything runs on your machine. No cloud APIs. No data leaves your device.

---

## How It Works

The core idea is that the best way to test understanding is to make the learner *teach*. GemmaGenius puts the student in the role of teacher: a confused 8-year-old named **Pip** asks them to explain a math concept. The student has to find the right words, and Pip reacts based on how close they are.

Two AI units share the same local model but serve completely different roles:

```
                    ┌─────────────────────────────────┐
                    │          MAIN MENU               │
                    │  [1] The Book Corner             │
                    │  [2] The Great Ground-Puller     │
                    │  [q] Quit                        │
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
                                          │  8-year-old         │ ──▶ streamed
                                          │  puzzle-solver      │     to terminal
                                          └─────────────────────┘
                                                      │
                                          session ends (mastery / exit)
                                                      │
                                                      ▼
                                             back to MAIN MENU
```

---

## Session Flow (Two-Phase Structure)

A complete session has two acts:

```
  MAIN MENU
  ─────────
  [1] The Book Corner
  [2] The Great Ground-Puller
  [q] Quit
        │ user picks a number
        ▼
PHASE 1 — ELICITATION
  Pip presents story_intro (verbatim from JSON, no Ollama call)
  → Student must explain the concept correctly (mastery)
        │
        ▼  (instant transition — no Ollama call)
  Pip celebrates with story_bridge, then immediately
  presents a DELIBERATE MISTAKE verbatim from JSON.
        │
        ▼
PHASE 2 — VERIFICATION BOSS FIGHT
  Pip presents verification_scenario (verbatim from JSON)
  → Student must catch and correct Pip's wrong application
        │
        ▼  (Phase 2 mastery)
  ★ TRUE MASTERY ACHIEVED ★  →  back to Main Menu

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
        └── anything else           →  "Session ended gracefully."  →  back to Main Menu
```

---

## Project Structure

```
Gemma4good/
├── main.py                       # Core engine — all logic lives here
├── knowledge.json                # Concept knowledge graph
├── Master Ingestion Prompt.txt   # LLM prompt for generating new concepts
├── Math_chapter1.md              # Example source curriculum (Grade 4 Geometry)
├── ARCHITECTURE.md               # Design rules and constraints
└── README.md                     # This file
```

---

## What Has Been Built (Layer 1)

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

### `main.py` — The Core Engine

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
| `mastery` | Student answered correctly for the current phase |
| `partial_hit` | Mentions shapes/lines/corners but geometric logic is incomplete. Pure material or theme descriptions (cement, cardboard) with no geometry are forced to `miss` |
| `miss` | Wrong answer, vague, or described only materials/themes without geometry |
| `question` | Student asked a clarifying question about the concept |
| `off_topic` | Joke, nonsense, or unrelated topic |
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
| `partial_hit` | none | Acknowledges the student's specific analogy, then immediately demands they describe the edges/lines of that object to force geometric reasoning |
| `miss` | `frustration++` | Makes a deliberately wrong binary guess so the student feels smart correcting her |
| `off_topic` | none | Laughs it off like a kid, immediately pivots back to the puzzle |
| `question` (1st or 2nd) | `clarification++` | Rephrases with simpler, different words |
| `question` (3rd+) | none | "Maybe we should look at a book together" |
| `give_up` | — | Reveals the full answer warmly, then asks if the student wants to retry. **Yes** → resets counters, re-prints the phase opening line, `continue`s. **No** → `return`s |

**Phase 2 routing:**

| Evaluator verdict | Counter effect | Pip's behaviour |
|---|---|---|
| `mastery` | — | Prints `★ TRUE MASTERY ACHIEVED ★` banner and breaks |
| `partial_hit` | none | Asks student to finish explaining why the roof idea is wrong |
| `miss` | none (no penalty) | Acts confused, asks if an L-shape really makes a flat roof |
| `off_topic` | none | Redirects back to the roof question |
| `question` (1st or 2nd) | `clarification++` | Rephrases the roof question without revealing the answer |
| `question` (3rd+) | none | Suggests drawing it out on paper together |
| `give_up` | — | Reveals the full answer warmly, then asks if the student wants to retry. **Yes** → resets counters, re-prints the verification scenario, `continue`s. **No** → `return`s |

**Recovery Loop detail (`give_up` path):**

1. One Ollama call — directive opens with `OVERRIDE FIREWALL:` to bypass the base persona's answer prohibition. Pip reveals the answer (pulled from `evaluator_ground_truth` or `verification_ground_truth` depending on phase) and ends with the verbatim question *"Now that you know the secret, do you want to try explaining it to me so we can finish?"*
2. A plain Python `input()` intercepts the reply — no AI needed for yes/no routing.
3. **Yes** (`yes / y / sure / ok / okay / yeah / i guess`) — resets `frustration_counter` and `clarification_counter` to 0, re-seeds `pips_last_question` with the correct phase opening line, prints it, and `continue`s back to the top of the loop.
4. **Anything else** — prints `[Session ended gracefully. You did a great job trying!]`, `return`s from `run_session`, and the outer menu loop shows the concept selector again.

**UX indicator:** `[Pip is thinking...]` is printed before every streaming Ollama call so the student knows a response is coming rather than seeing a silent pause.

**Turn 0** prints `story_intro` directly — no Ollama call, so the app starts instantly.

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com) installed and running locally
- A model pulled (default: `gemma4:e4b`)
- Python 3.10+
- `requests` library

```powershell
pip install requests
ollama pull gemma4:e4b
ollama serve
```

### Run

```powershell
cd "c:\Projects with Agents\Gemma4good"
python main.py
```

### Swap the model

Edit the `MODEL_NAME` constant at the top of `main.py`:

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

- **Layer 3:** Frustration escalation — use `frustration_counter` to trigger Pip's "hint ladder" (progressively stronger hints without giving the answer).
- **Layer 4:** Session scoring — report mastery rate, number of misses, and time-to-mastery at the end of each concept.
- **Layer 5:** Concept authoring CLI — add new concepts to `knowledge.json` interactively without editing the file by hand.
