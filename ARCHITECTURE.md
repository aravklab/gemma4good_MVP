# GemmaGenius: System Architecture & Rules

A local, privacy-first educational CLI app built on the Feynman Technique — the student teaches the AI, not the other way around.

---

## Core Rules (Non-Negotiable)

1. **No Agent Frameworks:** DO NOT use LangChain, LangGraph, AutoGen, or any OpenAI/Anthropic libraries.
2. **Tech Stack:** Raw Python 3.x only. Use `requests` to talk to Ollama at `http://localhost:11434/api/generate`.
3. **State Management:** Procedural `while` loops and plain Python variables. No async, no classes, no state machines.
4. **All inference is local:** Only Ollama models. No cloud API calls.

---

## Two Logical Units, One Model

Both units hit the same Ollama model but receive completely different system prompts:

| Unit | Role | Output |
|---|---|---|
| **Pip (The Actor)** | 8-year-old persona with an Information Firewall. Responds to the student. | Streamed natural language |
| **The Evaluator (The Logic)** | Silent background judge. Never seen by the user. | Strict JSON: `{"classification": "..."}` |

The Evaluator always runs first; its verdict determines the `state_directive` injected into Pip's prompt for that turn.

---

## Session State Variables

These are initialised at the start of every session in `run_session()`:

| Variable | Type | Purpose |
|---|---|---|
| `current_phase` | `int` (1 or 2) | Tracks Elicitation vs. Verification Boss Fight |
| `frustration_counter` | `int` | Increments on every `"miss"` (infrastructure for hint ladder) |
| `clarification_counter` | `int` | Increments on `"question"`; resets on any other classification |
| `pips_last_question` | `str` | Seeded for the Evaluator so it has question context each turn |

---

## Two-Phase Session Structure

### Phase 1 — Elicitation
Pip presents a story intro and asks the student to explain a concept. The Evaluator grades against `evaluator_ground_truth`.

- `mastery` → instant transition to Phase 2 (no API call; `story_bridge` + `verification_scenario` printed directly)
- `partial_hit` → Pip asks student to describe the geometry more precisely
- `miss` → Pip makes a deliberately wrong binary guess; `frustration_counter` increments
- `question` → Pip rephrases (max 2 times; then suggests looking at a book together)
- `off_topic` → Pip redirects to the puzzle
- `give_up` → **Escape Hatch** (see below)

### Phase 2 — Verification Boss Fight
Pip presents a scenario containing a deliberate mistake. The student must catch it and explain why it's wrong. The Evaluator grades against `verification_ground_truth`.

- `mastery` → `★ TRUE MASTERY ACHIEVED ★` banner; session ends
- All other classifications handled identically to Phase 1 (no `frustration_counter` increment)

---

## Escape Hatch & Recovery Loop

Triggered when the Evaluator returns `"give_up"`:

1. `state_directive` is set to `"OVERRIDE FIREWALL: ..."` — Pip reveals the answer verbatim from the ground truth.
2. Pip asks a strict Yes/No question: *"Now that you know the secret, do you want to try explaining it to me so we can finish?"*
3. A plain Python `input()` intercepts the reply (no AI call needed).
4. **Yes** → counters reset, phase re-seeded, session continues.
5. **No** → session ends gracefully; execution returns to the main menu.

---

## The Knowledge Graph (`knowledge.json`)

A local JSON file. No vector DB. No embeddings.

```
knowledge.json
└── concepts{}
    └── <CONCEPT_KEY>
        ├── name                      (display name)
        ├── secret_fact               (what Pip secretly knows)
        ├── goal                      (what the student must convey)
        ├── evaluator_ground_truth    (Phase 1 rubric)
        ├── story_bridge              (transition text at Phase 1 mastery)
        └── variants[]                (optional — enables Scenario Polymorphism)
            ├── variant_name
            ├── story_intro           (overrides root for this session)
            ├── verification_scenario (overrides root for this session)
            └── verification_ground_truth (Phase 2 rubric for this variant)
```

If a concept has a `variants` array, `main()` uses `random.choice()` to pick one variant per session, injecting its fields into `concept_data` before `run_session()` is called. Concepts without `variants` are fully backwards-compatible.

---

## Evaluator Classification States

| Value | Meaning |
|---|---|
| `mastery` | Student clearly explained the ground truth |
| `partial_hit` | Geometric logic mentioned but incomplete |
| `miss` | Wrong, confused, or no geometric explanation |
| `question` | Student is asking a clarifying question |
| `off_topic` | Joking, nonsense, or unrelated |
| `give_up` | Student explicitly surrenders |

---

## What Is NOT Built Yet (Planned)

- **Hint Ladder:** Use `frustration_counter` thresholds to escalate `state_directive` progressively (Level 1 → wrong guess, Level 2 → spatial hint, Level 3 → near-answer) before the Escape Hatch fires.
- **Session scoring / history**
- **New concept ingestion UI** (currently a manual prompt + paste workflow via `Master Ingestion Prompt.txt`)
