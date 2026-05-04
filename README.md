# GemmaGenius

A local, privacy-first educational app built on the **Feynman Technique** — the student explains a concept back to an AI persona, and the AI teaches by asking questions, not by giving answers.

Everything runs on your machine. No cloud APIs. No data leaves your device.

---

## Interfaces

| Interface | File | How to run |
|---|---|---|
| Web UI (Streamlit) — **recommended** | `app.py` | `streamlit run app.py` |
| CLI (original) | `main.py` | `python main.py` |
| Batch PDF ingestion | `ingest.py` | `python ingest.py <file.pdf>` |

Both `app.py` and `main.py` share the same `knowledge.json` knowledge graph and the same local Ollama model.

---

## How It Works

The core idea is that the best way to test understanding is to make the learner *teach*. GemmaGenius puts the student in the role of teacher: a confused AI persona asks them to explain a concept. The student must find the right words, and the persona reacts based on how close they are.

Two AI units share the same local model but serve completely different roles:

```
                    ┌──────────────────────────────────┐
                    │   WEB UI sidebar / CLI MENU       │
                    │  [1] The Book Corner              │
                    │  [2] The Earth's Invisible Tug   │
                    │  [3] Plants Are Secretly Heroes   │
                    └──────────────┬───────────────────┘
                                   │ concept selected
                                   ▼
Student types an answer
        │
        ▼
┌─────────────────┐   JSON verdict          ┌──────────────────┐
│   The Evaluator │ ──────────────────────▶ │  Routing Logic   │
│  (silent judge) │  mastery / miss /       │  (plain Python)  │
└─────────────────┘  partial_hit /          └────────┬─────────┘
                     question /                      │ state_directive
                     off_topic /                     ▼
                     give_up             ┌─────────────────────┐
                                         │  Persona (Actor)    │
                                         │  Pip / Alex / Riley │──▶ chat bubble
                                         └─────────────────────┘
                                                     │
                                         session ends (mastery / exit)
                                                     │
                                                     ▼
                                         back to concept selector
```

---

## Session Flow (Two-Phase Structure)

```
  CONCEPT SELECTOR
  ────────────────
  Pick a concept
        │
        ▼
PHASE 1 — ELICITATION
  Persona presents story_intro (from JSON, no Ollama call for Turn 0)
  → Student must explain the concept correctly (mastery)
        │
        ▼  (seamless transition — single Ollama call)
  Persona validates the student, then IMMEDIATELY presents
  a DELIBERATE MISTAKE from the verification_scenario.
        │
        ▼
PHASE 2 — VERIFICATION BOSS FIGHT
  → Student must catch and correct the persona's wrong application
        │
        ▼  (Phase 2 mastery)
  ★ TRUE MASTERY ACHIEVED ★  →  achievement saved  →  back to selector

  ─── At any point ───
  Student gives up
        │
        ▼  (give_up — OVERRIDE FIREWALL)
  Persona reveals the answer, asks: "Want to try explaining it to me?"
        ├── yes  →  counters reset, puzzle replays from current phase
        └── no   →  "Session ended gracefully."
```

---

## Project Structure

```
Gemma4good/
├── app.py                  # Web UI — Streamlit (Kid Mode + Parent Dashboard)
├── main.py                 # CLI engine — original interface, untouched
├── ingest.py               # Batch PDF → knowledge.json ingestion pipeline
├── knowledge.json          # Concept knowledge graph (shared by all interfaces)
├── student_profile.json    # Persistent student achievements (auto-created)
├── ARCHITECTURE.md         # Design rules and constraints
└── README.md               # This file
```

---

## What Has Been Built

### `knowledge.json` — The Knowledge Graph

A structured JSON file that stores teachable concepts. Each concept contains:

| Field | Purpose |
|---|---|
| `name` | Display name shown in selector and headers |
| `story_bridge` | Persona's Phase 1 win reaction |
| `secret_fact` | The answer the persona secretly knows but must never say |
| `goal` | Plain-English description of what the persona is nudging toward |
| `evaluator_ground_truth` | Rubric the Evaluator grades against in Phase 1 |
| `story_intro` | *(root-level)* Persona's opening line if no `variants` array |
| `verification_scenario` | *(root-level)* Persona's deliberate-mistake question if no `variants` |
| `verification_ground_truth` | *(root-level)* Phase 2 rubric if no `variants` |
| `variants` | *(optional)* Array of randomised scenario objects |
| `persona_config` | *(optional, injected by ingest.py)* Persona identity and voice |

**Scenario Polymorphism:** When a concept has a `"variants"` key, the engine picks one at random each session using `random.choice()` and injects that variant's `story_intro`, `verification_scenario`, and `verification_ground_truth` into the working copy. Root-level fields are shared across all variants.

**Current concepts: 3**

| Key | Name | Variants | Source |
|---|---|---|---|
| `C1_Right_Angle` | The Book Corner | — | Hand-authored |
| `C3_New_Concept` | The Earth's Invisible Tug | 3 | Hand-authored |
| `C3_Plants_Aren't_Just_Sitting_There!` | Plants Are Secretly Superheroes | 2 | PDF-ingested |

---

### `app.py` — The Streamlit Web UI

A full-featured web application with two views selectable from the sidebar.

#### Dual-View Navigation

| View | Who uses it | What it shows |
|---|---|---|
| 🎮 Play (Kid Mode) | Student | Pip's chat interface |
| 📊 Dashboard (Parent Mode) | Parent | Analytics, PDF ingestion, concept review |

Switching views **never clears the active chat session** — the student's game is preserved if a parent checks the dashboard mid-session.

#### Persistence Layer

- `student_profile.json` — loaded once per browser session into `st.session_state.profile`. Tracks per-concept achievements: mastery status, total frustration triggers, and boss-fight attempt count.
- On every Phase 2 win, the achievement is written to disk immediately via `save_profile()`.
- The **Trophy Room** in the sidebar shows a gold star for every mastered concept.

#### Session State (replaces `main.py`'s `while` loop)

| Key | Default | Purpose |
|---|---|---|
| `messages` | `[]` | Full chat history |
| `current_phase` | `1` | 1 = Elicitation, 2 = Boss Fight |
| `frustration_counter` | `0` | Phase 1 miss count |
| `clarification_counter` | `0` | Resets on any non-question turn |
| `boss_fight_attempts` | `0` | Phase 2 submission count |
| `concept_data` | `None` | Active concept dict |
| `pips_last_question` | `""` | Fed to the Evaluator each turn |
| `awaiting_retry` | `False` | True after give_up — shows Yes/No buttons |
| `pending_levels` | `[]` | AI-generated concepts awaiting parent review |
| `profile` | loaded | Student achievements |
| `last_prompt` | `""` | Debug: last prompt sent to Ollama |
| `last_response` | `""` | Debug: last raw response from Ollama |
| `last_classification` | `"None"` | Debug: last evaluator verdict |
| `last_evaluator_rationale` | `"None"` | Debug: raw evaluator JSON before parsing |

#### Routing — The Bridging Constraint (Phase 1 Miss)

When the Evaluator returns `miss` in Phase 1, the persona no longer makes a generic wrong guess. Instead it receives a **Bridging Constraint** directive that injects:
1. The exact `story_intro` text (the original puzzle)
2. The student's last message (verbatim)

The persona is forced to acknowledge what the student said, explain why it doesn't solve *that specific puzzle*, and re-ask. This eliminates topic drift on incorrect answers.

#### Dynamic Persona Avatar

`get_persona_avatar()` reads `concept_data.persona_config.avatar_emoji` if present, and falls back to `"👧🏼"` (Pip) for legacy hand-authored concepts. The avatar updates automatically when the active concept changes.

#### Parent Dashboard

| Section | What it shows |
|---|---|
| Summary metrics | Concepts Mastered, Learning Friction (total frustration triggers) |
| Concept Breakdown | `st.dataframe` with status, frustration triggers, boss-fight attempts per concept |
| Suggested Activity | Home activity for the concept with the highest friction |
| Concept Review Queue | AI-generated concepts pending parent approval — Add to Game or Discard |
| PDF Ingestion | Upload a PDF → fail-fast guardrail (500 char min) → AI generates concept → lands in review queue |

#### Developer Console

Toggle **🐛 Enable Debug Mode** in the Developer Settings expander at the bottom of the sidebar. Reveals:

- **🧠 Background Evaluator** — the classification verdict and the raw JSON the model returned
- **👧🏼 Pip Generation** — the full system + user prompt sent to Pip and Pip's raw output
- **💾 Session State** — live JSON dump of all counters, phase, and concept data

---

### `ingest.py` — Batch PDF Ingestion Pipeline

A standalone CLI script that reads a PDF, chunks it into paragraph blocks, and generates new `knowledge.json` concepts via Ollama.

```powershell
# Preview output without writing anything
python ingest.py notes.pdf --dry-run

# Process at most 3 chunks with a specific model
python ingest.py textbook.pdf --model gemma4:e4b --chunk-limit 3

# Full ingestion
python ingest.py chapter.pdf
```

**Options:**

| Flag | Default | Purpose |
|---|---|---|
| `--dry-run` | off | Print JSON to terminal, do not write to file |
| `--model NAME` | `gemma4:e4b` | Ollama model to use |
| `--knowledge PATH` | `knowledge.json` | Target knowledge file |
| `--max-chars N` | `3000` | Max characters per chunk |
| `--chunk-limit N` | all | Process at most N chunks |

**Safety features:**
- Duplicate detection (case-insensitive name match against existing concepts)
- JSON parse error recovery (strips markdown fences, handles `{"skip": true}` signal)
- Connection error aborts cleanly; timeout skips the chunk and continues

#### Dynamic Persona System

The ingestion pipeline automatically analyses the complexity of the source text and selects the appropriate persona:

| Complexity | Persona | Age | Voice |
|---|---|---|---|
| Primary | **Pip** 🧒 | 8 | Curious, uses toy and playground analogies |
| Intermediate | **Alex** 🧑‍🏫 | 13 | Slightly skeptical, uses sports and social analogies |
| Advanced | **Riley** 🕵️ | 16 | Overzealous detective who invents wild, confidently wrong theories and demands the student confirm or debunk them |

Each ingested concept includes a `persona_config` block:

```json
{
  "name": "Pip",
  "age": 8,
  "complexity_level": "Primary",
  "avatar_emoji": "🧒",
  "voice_tone": "Curious 8-year-old who uses toy and playground analogies"
}
```

Riley-specific rule: verification scenarios must be confidently wrong wild theories (not simple questions), ending with *"I've cracked the code, haven't I?"*

---

### Classification States (Evaluator)

| State | Meaning |
|---|---|
| `mastery` | Student clearly explained the core mechanism (childlike language is fine) |
| `partial_hit` | Relevant ideas mentioned but explanation is incomplete or missing the core mechanism |
| `miss` | Wrong, confused, guessing, or parroting vocabulary without explaining the mechanism |
| `question` | Student asked a clarifying question |
| `off_topic` | Joke, nonsense, or completely unrelated topic |
| `give_up` | Explicit surrender ("I don't know", "just tell me"). A wrong answer is still `miss` |

---

### Phase 1 Routing

| Verdict | Counter effect | Persona behaviour |
|---|---|---|
| `mastery` | — | OVERRIDE FIREWALL: validate + immediately present boss scenario |
| `partial_hit` | none | Validate what they got right, ask guiding question |
| `miss` | `frustration++` | **Bridging Constraint**: acknowledge what they said, explain why it doesn't fix the puzzle, ask again |
| `off_topic` | none | Acknowledge, re-state exact `story_intro` verbatim |
| `question` (1–2) | `clarification++` | Answer briefly, re-state exact `story_intro` verbatim |
| `question` (3+) | none | "Maybe we should look at a book together" |
| `give_up` | — | OVERRIDE FIREWALL: reveal answer using `secret_fact`, ask Yes/No retry |

### Phase 2 Routing

| Verdict | Counter effect | Persona behaviour |
|---|---|---|
| `mastery` | — | ★ TRUE MASTERY ACHIEVED ★ + save achievement |
| `partial_hit` | none | Validate partial logic, ask follow-up to fully debunk the scenario |
| `miss` | none | Act confused, nudge toward the flaw in the scenario |
| `off_topic` | none | Acknowledge, re-state exact `verification_scenario` verbatim |
| `question` (1–2) | `clarification++` | Answer briefly, re-state exact `verification_scenario` verbatim |
| `question` (3+) | none | "Maybe we should draw it out on paper" |
| `give_up` | — | OVERRIDE FIREWALL: reveal answer using `verification_ground_truth`, ask Yes/No retry |

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com) installed and running locally
- A model pulled (default: `gemma4:e4b`)
- Python 3.10+

```powershell
pip install requests streamlit pypdf
ollama pull gemma4:e4b
ollama serve
```

### Run — Web UI (recommended)

```powershell
cd "C:\Projects with Agents\Gemma4good"
streamlit run app.py
```

### Run — CLI

```powershell
cd "C:\Projects with Agents\Gemma4good"
python main.py
```

### Run — PDF Batch Ingestion

```powershell
cd "C:\Projects with Agents\Gemma4good"
python ingest.py path\to\your\textbook.pdf --dry-run
```

### Swap the model

Edit `MODEL_NAME` at the top of `app.py` or `main.py`, and `DEFAULT_MODEL` in `ingest.py`:

```python
MODEL_NAME = "gemma4:e4b"   # any model you have pulled locally
```

---

## Architecture Constraints

Defined in `ARCHITECTURE.md` — enforced throughout:

- **No agent frameworks** — no LangChain, LangGraph, AutoGen, or OpenAI SDK.
- **Raw Python + `requests` only** — one dependency for all Ollama HTTP calls.
- **Procedural state** — `session_state` keys and plain variables; no state machine libraries.
- **Two logical units, one model** — Pip/persona and the Evaluator are prompt roles, not separate processes.
- **Local knowledge graph** — flat JSON file; no vector DB, no embeddings.
- **Privacy-first** — no data leaves the device; all inference runs on local Ollama.

---

## Adding New Concepts

### Via PDF (recommended)

```powershell
python ingest.py your_material.pdf --dry-run   # preview first
python ingest.py your_material.pdf             # write to knowledge.json
```

Or use the **Parent Dashboard** in the web UI: upload a PDF, review the generated concept card, and click **✅ Add to Game**.

### Manual Authoring Rules

Add each concept as a direct key inside `"concepts"`:

```json
"concepts": {
  "C1_Right_Angle": { ... },
  "C4_My_New_Concept": { ... }
}
```

Validate after editing:

```powershell
python -c "import json; d=json.load(open('knowledge.json')); print(list(d['concepts'].keys()))"
```

---

## What's Next (Planned Layers)

- **Layer 3 — Hint Ladder:** Use `frustration_counter` to trigger progressively stronger hints without giving the answer away.
- **Layer 4 — Session Scoring:** Report mastery rate, misses, and time-to-mastery at the end of each concept; surface trends in the Parent Dashboard.
- **Layer 5 — Persona Voice in Pip Prompt:** Propagate `persona_config.voice_tone` into `compile_pip_prompt` so Alex and Riley's voices are enforced at the system-prompt level, not just at ingestion time.
- **Layer 6 — Multi-student Profiles:** Support multiple named profiles in `student_profile.json` with a profile switcher in the sidebar.
- **Layer 7 — Frustration Alerts:** Parent Dashboard notification when a concept's friction score crosses a threshold, with an email or push nudge.
