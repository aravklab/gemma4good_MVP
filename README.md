# GemmaGenius

A local, privacy-first educational app built on the **Feynman Technique** — the student explains a concept back to an AI persona, and the AI teaches by asking questions, not by giving answers.

Everything runs on your machine. No cloud APIs. No data leaves your device.

---

## Quick Start

```powershell
cd "C:\Projects with Agents\Gemma4good"

# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the required Ollama models
ollama pull gemma4:e4b
ollama pull nomic-embed-text
ollama serve        # keep this running in a separate terminal

# 3. Launch the web app
streamlit run app.py
```

---

## Interfaces

| Interface | File | How to run |
|---|---|---|
| Web UI (Streamlit) — **recommended** | `app.py` | `streamlit run app.py` |
| CLI (original / legacy) | `main.py` | `python main.py` |
| Batch PDF ingestion | `ingest.py` | `python ingest.py <file.pdf>` |

- `app.py` reads approved concepts from **ChromaDB** (local vector store) and generates levels Just-In-Time.
- `main.py` is the original terminal interface; still reads from `knowledge.json`.
- `ingest.py` processes PDFs and writes skeletons to ChromaDB.

---

## How It Works

The core idea is that the best way to test understanding is to make the learner *teach*. GemmaGenius puts the student in the role of teacher: a confused AI persona asks them to explain a concept. The student must find the right words, and the persona reacts based on how close they are.

Two AI units share the same local Ollama model but serve completely different roles:

```
                    ┌──────────────────────────────────┐
                    │   SIDEBAR — Subject Folders       │
                    │  📁 Biology                       │
                    │    ▶ Photosynthesis               │
                    │    🔒 Osmosis                     │
                    │  📁 General                       │
                    │    ⭐ Right Angles (mastered)     │
                    └──────────────┬───────────────────┘
                                   │ concept clicked
                                   ▼
                         generate_level_jit()
                    (if skeleton — Ollama generates
                     fresh story + persona + boss fight)
                                   │
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
```

---

## Session Flow — Two-Phase Structure

```
  CONCEPT SELECTOR (sidebar)
  ──────────────────────────
  Click an unlocked concept
        │
        ▼
  generate_level_jit()  ← (skeleton concepts only)
  Creates a FRESH story, persona, boss fight on every click
        │
        ▼
PHASE 1 — ELICITATION
  Persona presents story_intro (no Ollama call for Turn 0)
  → Student must explain the concept correctly
        │
        ▼  (seamless transition — single Ollama call)
  Persona validates the student, then IMMEDIATELY presents
  a DELIBERATE MISTAKE from verification_scenario.
        │
        ▼
PHASE 2 — BOSS FIGHT (Verification)
  → Student must catch and correct the persona's wrong application
        │
        ▼  (Phase 2 mastery)
  ★ TRUE MASTERY ACHIEVED ★  →  achievement saved  →  concept shows ⭐

  ─── At any point ───
  Student gives up
        │
        ▼  (give_up — OVERRIDE FIREWALL)
  Persona reveals the answer, asks: "Want to try explaining it to me?"
        ├── yes  →  counters reset, puzzle replays from current phase
        └── no   →  "Session ended gracefully."
```

---

## JIT Architecture (Just-In-Time Level Generation)

GemmaGenius separates **ingestion** from **story generation**:

```
INGESTION TIME (ingest.py)        PLAY TIME (app.py)
──────────────────────────        ──────────────────
PDF → semantic chunks             Student clicks concept
    → LLM extracts skeleton:          │
      {concept_name,                  ▼
       ground_truth_logic}        generate_level_jit()
    → nomic-embed-text                │  Ollama generates:
    → ChromaDB upsert                 │  persona_config
      status: "pending"               │  story_intro
                                      │  verification_scenario
Parent approves in Dashboard          │  boss_fight_logic
    → status: "approved"              │  home_activity
    → visible to student              │
                                      ▼
                                  start_session(merged_concept)
                                  → FRESH story every click
```

**Why JIT?**
- Ingestion is fast — no story generation overhead per chunk
- Every session is unique — the same concept generates a new story each click
- Reduces LLM context pressure during ingestion, improving skeleton quality

---

## Project Structure

```
Gemma4good/
├── app.py                      # Web UI — Streamlit (Kid Mode + Parent Dashboard)
├── main.py                     # CLI engine — original interface (legacy, knowledge.json only)
├── ingest.py                   # Batch PDF → ChromaDB skeleton ingestion pipeline
├── knowledge.json              # Legacy concept graph (used by main.py + app.py as fallback)
├── student_profile.json        # Persistent student achievements (auto-created on first win)
├── rejected_telemetry.jsonl    # Dead Letter Queue — append-only rejected concept log
├── requirements.txt            # Direct Python dependencies (pinned)
├── chroma_db/                  # Local ChromaDB vector store (gitignored, auto-created)
├── LEARNINGS.md                # Architectural and technical lessons from building this
└── README.md                   # This file
```

---

## app.py — The Streamlit Web UI

### Dual-View Navigation

| View | Who uses it | What it shows |
|---|---|---|
| 🎮 Play (Kid Mode) | Student | Persona chat + subject sidebar |
| 📊 Dashboard (Parent Mode) | Parent | Analytics, PDF ingestion, concept review, curriculum management |

Switching views **never clears the active chat session** — the student's game is preserved if a parent checks the dashboard mid-session.

### Kid Mode — Subject Folders + Linear Path

Concepts are grouped into subject folders in the sidebar (e.g., 📁 Biology, 📁 Mathematics). Within each folder, concepts are sorted by `sequence_order` (a float, explained below).

**Look-ahead Locking (buffer = 3):**

```
  Pass 1: find max_mastered_seq — the highest sequence_order the student has mastered
  Pass 2: for each concept in the folder:
            if is_mastered                → ⭐ (always clickable)
            if seq <= max_mastered + 3    → ▶ (unlocked, clickable)
            if seq == 999                 → ○ (legacy, always unlocked)
            else                          → 🔒 (locked, button disabled)
```

This lets students work on up to 3 concepts simultaneously while preventing them from skipping ahead.

### Dynamic Persona System

The persona is stored inside every concept's `persona_config` block and drives the entire UI:

| Persona | Age | Avatar | Voice |
|---|---|---|---|
| **Pip** | 8 | 👧🏼 | Curious, uses toy and playground analogies |
| **Alex** | 13 | 👦🏽 | Slightly skeptical, uses sports/social/allowance analogies |
| **Riley** | 16 | 🕵️ | Overzealous detective — invents wild, confidently wrong theories |

The Play Mode subheader, chat avatar, win message, chat placeholder, and error fallbacks all update automatically to reflect the active concept's persona.

### Persistence Layer

- `student_profile.json` — loaded once per session into `st.session_state.profile`.
- Achievements are keyed by the **stable slug ID** (e.g. `right_angle`), not the display name. This prevents duplicates if a concept's name changes.
- On Phase 2 win, the achievement is written to disk immediately via `save_profile()`.

### Parent Dashboard Sections

| Section | What it does |
|---|---|
| Summary Metrics | Concepts Mastered + Learning Friction (total frustration triggers) |
| Concept Breakdown | DataFrame with per-concept status, frustration triggers, boss-fight attempts |
| Suggested Activity | Reads `home_activity` from the highest-friction concept's JSON — no hardcoded lookup |
| Review Queue | ChromaDB query for `status: pending`; Approve (sets `approved`) or Reject & Delete (logs + deletes vector) |
| PDF Upload | Assigns subject + uploads PDF → AI extracts skeleton concepts → lands in Review Queue |
| 🔧 Repair Sequence Numbers | Assigns proper 1-based sequence numbers to any concepts still at `sequence_order=999` |
| 📋 Active Curriculum | Lists all approved concepts; "Remove" button deletes from ChromaDB and preserves achievement name |

### Routing — Bridging Constraint (Phase 1 Miss)

When the Evaluator returns `miss`, the persona receives a **Bridging Constraint** directive that injects:
1. The exact `story_intro` text (the original puzzle)
2. The student's last message (verbatim)

The persona must acknowledge what the student said, explain why it doesn't fix *that specific puzzle*, and ask again. This eliminates topic drift on incorrect answers.

### Dead Letter Queue (DLQ)

When a parent rejects a concept:
1. The full concept JSON + rejection reason append to `rejected_telemetry.jsonl` (safe to `jq` / `pandas`).
2. The vector is physically deleted from ChromaDB to prevent context pollution.

### Developer Console

Toggle **🐛 Enable Debug Mode** in the sidebar. Reveals:

- **🧠 Background Evaluator** — classification verdict and raw JSON
- **👧🏼 Persona Generation** — full prompt sent and raw LLM output
- **💾 Session State** — live JSON dump of all counters, phase, and concept data

---

## ingest.py — Batch PDF Ingestion Pipeline

Standalone CLI script. Reads a PDF → semantic chunks with overlap → skeleton extraction → ChromaDB upsert.

```powershell
# Preview without writing
python ingest.py notes.pdf --dry-run

# Full ingestion: Biology, Chapter 2
python ingest.py chapter2.pdf --subject Biology --module 2

# Limit chunks during development
python ingest.py notes.pdf --chunk-limit 3 --dry-run
```

### CLI Options

| Flag | Default | Purpose |
|---|---|---|
| `--dry-run` | off | Print JSON to terminal, do not write to ChromaDB |
| `--model NAME` | `gemma4:e4b` | Ollama model for concept extraction |
| `--embed-model NAME` | `nomic-embed-text` | Ollama model for embeddings |
| `--chroma-path PATH` | `./chroma_db` | Local ChromaDB persistence directory |
| `--collection NAME` | `curriculum` | ChromaDB collection name |
| `--max-chars N` | `3000` | Max characters per semantic chunk |
| `--overlap N` | `300` | Character overlap between adjacent chunks |
| `--chunk-limit N` | all | Process at most N chunks |
| `--subject NAME` | `General` | Subject category tag applied to all concepts |
| `--module N` | `1` | Module number for fractional sequencing |

### Skeleton Schema (what ingest.py stores)

```json
{
  "concept_name": "Photosynthesis Energy Capture",
  "ground_truth_logic": "Plants convert light energy into glucose using CO2 and water...",
  "subject": "Biology",
  "sequence_order": 2.1,
  "status": "pending"
}
```

The `story_intro`, `persona_config`, `verification_scenario`, `boss_fight_logic`, and `home_activity` are **NOT** stored at ingestion time — they are generated by `generate_level_jit()` in `app.py` on every concept click.

### Fractional Sequencing

```
--module 1  →  concepts numbered  1.1, 1.2, 1.3 …
--module 2  →  concepts numbered  2.1, 2.2, 2.3 …
```

Multiple chapters can be ingested independently without renumbering. The Parent Dashboard's "Repair Sequence Numbers" tool can fix legacy concepts still at `sequence_order=999`.

### Safety Features

- Semantic chunking with configurable overlap — avoids cutting concepts mid-paragraph
- Deterministic IDs (slugified concept name) — `upsert` prevents duplicates on re-ingestion
- JSON parse error recovery — strips markdown fences, handles `{"skip": true}` signal
- All concepts start `status: "pending"` — nothing enters gameplay without parent approval

---

## ChromaDB Concept Lifecycle

```
  ingest.py / in-app PDF upload
        │
        ▼
  status: "pending"   ←── visible in Parent Dashboard Review Queue
        │
        ├── ✅ Approve    →  status: "approved"  ←── visible to students
        │
        └── 🗑️ Reject     →  logged to rejected_telemetry.jsonl (DLQ)
                              + physically deleted from ChromaDB
```

Concepts from `knowledge.json` are loaded as a legacy fallback and merged with ChromaDB concepts in `app.py`'s `main()` function. ChromaDB concepts take precedence on name collision.

---

## Evaluator Classification States

| State | Meaning |
|---|---|
| `mastery` | Student clearly explained the core mechanism (simple language is fine) |
| `partial_hit` | Relevant ideas present but explanation is incomplete |
| `miss` | Wrong, confused, guessing, or parroting vocabulary without explaining mechanism |
| `question` | Student asked a clarifying question or requested a hint |
| `off_topic` | Joke, nonsense, or completely unrelated topic |
| `give_up` | Unambiguous, explicit surrender — a wrong answer is still `miss` |

---

## Phase 1 Routing

| Verdict | Counter effect | Persona behaviour |
|---|---|---|
| `mastery` | — | OVERRIDE FIREWALL: validate + immediately present boss scenario |
| `partial_hit` | none | Validate what they got right, ask guiding question |
| `miss` | `frustration++` | **Bridging Constraint**: anchor to original puzzle, ask again |
| `off_topic` | none | Acknowledge, re-state exact `story_intro` verbatim |
| `question` (1–2) | `clarification++` | Answer briefly, re-state exact `story_intro` verbatim |
| `question` (3+) | none | "Maybe we should look at a book together" |
| `give_up` | — | OVERRIDE FIREWALL: reveal answer, ask Yes/No retry |

## Phase 2 Routing

| Verdict | Counter effect | Persona behaviour |
|---|---|---|
| `mastery` | — | ★ TRUE MASTERY ACHIEVED ★ + save achievement |
| `partial_hit` | none | Validate partial logic, ask follow-up |
| `miss` | none | Act confused, nudge toward the flaw in the scenario |
| `off_topic` | none | Acknowledge, re-state exact `verification_scenario` verbatim |
| `question` (1–2) | `clarification++` | Answer briefly, re-state scenario verbatim |
| `question` (3+) | none | "Maybe we should draw it out on paper" |
| `give_up` | — | OVERRIDE FIREWALL: reveal answer, ask Yes/No retry |

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com) installed and running locally
- Python 3.10+

```powershell
# Install Python dependencies
pip install -r requirements.txt

# Pull Ollama models (one-time)
ollama pull gemma4:e4b
ollama pull nomic-embed-text

# Keep Ollama running
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

### Swap the Model

Edit `MODEL_NAME` at the top of `app.py` or `main.py`, and `DEFAULT_MODEL` in `ingest.py`:

```python
MODEL_NAME = "gemma4:e4b"   # any model you have pulled locally
```

---

## Architecture Constraints

- **No agent frameworks** — no LangChain, LangGraph, AutoGen, or OpenAI SDK.
- **Raw Python + `requests` only** — one dependency for all Ollama HTTP calls.
- **Procedural state** — `session_state` keys and plain variables; no state machine libraries.
- **Two logical units, one model** — Persona and Evaluator are prompt roles, not separate processes.
- **Local vector store** — ChromaDB with `nomic-embed-text` embeddings via Ollama; no external DB.
- **Privacy-first** — no data leaves the device; all inference and embedding runs on local Ollama.

---

## Adding New Concepts

### Via PDF — In-App (recommended)

1. Open **Parent Dashboard** → scroll to "Generate Concept from PDF"
2. Select a subject from the dropdown
3. Upload a PDF — the AI extracts one skeleton per major section
4. Review each concept in the **Review Queue** and click ✅ Approve

### Via PDF — Batch CLI

```powershell
python ingest.py your_material.pdf --dry-run          # preview first
python ingest.py your_material.pdf --subject Math     # write to ChromaDB
```

Then open the Parent Dashboard to review and approve.

### Manual Authoring

Add concepts directly to `knowledge.json` for `main.py` (CLI). For the web UI, all concepts must pass through ChromaDB — use the in-app uploader or `ingest.py`.

---

## What's Next (Planned Layers)

- **Hint Ladder:** Use `frustration_counter` to trigger progressively stronger hints without giving the answer away.
- **Session Scoring:** Report mastery rate, misses, and time-to-mastery at end of each concept; surface trends in the Parent Dashboard.
- **Multi-student Profiles:** Support multiple named profiles in `student_profile.json` with a profile switcher in the sidebar.
- **Frustration Alerts:** Parent Dashboard notification when a concept's friction score crosses a threshold.
- **Semantic Concept Search:** Use ChromaDB's vector search to surface related concepts when a student struggles, enabling adaptive learning paths.
- **JIT Story Caching:** Optionally cache the JIT-generated story on first play so the same story persists for a given student across sessions while remaining regeneratable on demand.
