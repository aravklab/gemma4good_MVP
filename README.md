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

# 3. (Optional) Download the local Whisper model for voice input
python scripts/download_whisper_model.py

# 4. Launch the web app
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
- `main.py` is the original terminal interface; reads from `knowledge.json`.
- `ingest.py` processes PDFs and writes skeletons to ChromaDB.

---

## How It Works

Two AI units share the same local Ollama model but serve completely different roles:

```
                    ┌──────────────────────────────────┐
                    │   SIDEBAR — Subject Folders       │
                    │  📁 Biology                       │
                    │    ▶ Photosynthesis               │
                    │    🔄 Osmosis  (retry unlocked)   │
                    │    🆘 Mitosis  (needs help)       │
                    │    🔒 Genetics (locked)           │
                    │  📁 General                       │
                    │    ⭐ Right Angles (mastered)     │
                    └──────────────┬───────────────────┘
                                   │ concept clicked
                                   ▼
                         generate_level_jit()
                    (Ollama generates fresh story,
                     persona, boss fight — or injects
                     narrative continuity on retry)
                                   │
                                   ▼
Student submits answer (typed or spoken)
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
  generate_level_jit()
  → Fresh story + random setting every click
  → If concept has a resolved struggle, Pip opens with
    narrative continuity ("Remember how we got stuck on this?")
        │
        ▼
PHASE 1 — ELICITATION
  Persona presents story_intro
  → Student must explain the concept correctly

  FRUSTRATION LADDER (on each miss):
    Miss 1: Bridging Constraint — anchor to puzzle, ask again
    Miss 2: Disguised mega-hint — persona voices growing doubt
    Miss 3+: Aha! rescue — persona models the answer, student echoes it back
        │
        ▼  (mastery — seamless single Ollama call transition)
  Persona validates, immediately presents a DELIBERATE MISTAKE
        │
        ▼
PHASE 2 — BOSS FIGHT (Verification)
  → Student must catch and correct the persona's wrong application
  (same frustration ladder applies)
        │
        ▼  (Phase 2 mastery)
  ⭐ TRUE MASTERY ACHIEVED  →  achievement saved  →  balloon animation

  ─── At any point ───────────────────────────────────────────────
  Student gives up OR clarification loop hits threshold
        │
        ▼
  Persona explains answer (OVERRIDE FIREWALL), then:
  ┌──────────────────────────────────────────────────────────────┐
  │  ✅ Yes, try again!  →  counters reset, puzzle replays        │
  │  ❌ No, I'm done     →  session ends gracefully              │
  │                         concept flagged 🆘 in profile         │
  └──────────────────────────────────────────────────────────────┘
```

---

## The Student Struggle State Machine

GemmaGenius tracks three states per concept in `student_profile.json`:

| Sidebar icon | State | What happened |
|---|---|---|
| ○ / ▶ | Untouched | Never attempted |
| ⭐ | Mastered | Phase 2 win — achievement saved |
| 🆘 | Needs Help (active) | Abandoned — locked until parent intervenes |
| 🔄 | Resolved | Parent unlocked after intervention — Pip will open with narrative continuity |

**Parent Dashboard flow:**
1. Parent sees the **🆘 Needs Help** section — exit reason, attempt count, suggested home activity
2. Parent does the activity IRL with the child
3. Parent clicks **🔓 Unlock** — concept moves to `resolved` state
4. Child clicks **🔄** in the sidebar — Pip opens: *"Hey! Remember how we got stuck on this before? I've been thinking about it..."*
5. If the child masters it → `🔄 Comeback!` badge in the achievements table
6. If the child fails again → concept re-locks to 🆘 with incremented attempt count

The parent can also **🔒 Re-lock** a resolved concept if the child needs more preparation before retrying.

---

## JIT Architecture (Just-In-Time Level Generation)

GemmaGenius separates **ingestion** (fast skeleton extraction) from **story generation** (rich, dynamic, on-demand):

```
INGESTION TIME (ingest.py)        PLAY TIME (app.py)
──────────────────────────        ──────────────────
PDF → semantic chunks             Student clicks concept
    → LLM extracts skeleton:          │
      {concept_name,                  ▼
       ground_truth_logic}        generate_level_jit(
    → nomic-embed-text                  skeleton,
    → ChromaDB upsert                   retry_context  ← if resolved struggle
      status: "pending"             )
                                          │
Parent approves in Dashboard              │  Ollama generates:
    → status: "approved"                  │  persona_config (Pip/Alex/Riley)
    → visible to student                  │  story_intro + random setting seed
                                          │  verification_scenario
                                          │  boss_fight_logic
                                          │  home_activity
                                          ▼
                                  start_session(merged_concept)
                                  → FRESH story every click
```

**Why JIT?**
- Ingestion is fast — no story generation overhead per chunk
- Every click generates a fresh story with a random real-world setting (playground, birthday party, science fair, camping trip, etc.) — the LLM cannot repeat itself
- Retry sessions inject a **Narrative Continuity Directive** so Pip acknowledges the previous struggle rather than pretending it never happened
- Only pure legacy `knowledge.json` concepts use a static pre-written story

---

## Project Structure

```
Gemma4good/
├── app.py                      # Web UI — Streamlit (Kid Mode + Parent Dashboard)
├── main.py                     # CLI engine — original interface (legacy, knowledge.json only)
├── ingest.py                   # Batch PDF → ChromaDB skeleton ingestion pipeline
├── knowledge.json              # Legacy concept graph (main.py + app.py fallback)
├── student_profile.json        # Persistent student achievements + struggle records
├── rejected_telemetry.jsonl    # Dead Letter Queue — append-only rejected concept log
├── requirements.txt            # Direct Python dependencies
├── ARCHITECTURE.md             # Full system architecture, state machines, data flows
├── LEARNINGS.md                # Hard-won technical lessons from building this
├── scripts/
│   └── download_whisper_model.py   # One-time Whisper model download (SSL-proxy safe)
├── whisper_model/
│   └── base/                   # Local Whisper model bundle (gitignored, ~145 MB)
└── chroma_db/                  # Local ChromaDB vector store (gitignored, auto-created)
```

---

## app.py — The Streamlit Web UI

### Dual-View Navigation

| View | Who uses it | What it shows |
|---|---|---|
| 🎮 Play (Kid Mode) | Student | Persona chat + subject sidebar + audio controls |
| 📊 Dashboard (Parent Mode) | Parent | Analytics, PDF ingestion, concept review, curriculum management |

Switching views **never clears the active chat session** — the student's game is preserved if a parent checks the dashboard mid-session.

### Kid Mode — Subject Folders + Linear Path

Concepts are grouped into subject folders (e.g., 📁 Biology, 📁 Mathematics). Within each folder they are sorted by `sequence_order`.

**Look-ahead Locking (buffer = 3):**

```
Pass 1: find max_mastered_seq — highest sequence_order the student has mastered
Pass 2: for each concept in the folder:
          if is_mastered                  → ⭐  (always clickable)
          if needs_help active            → 🆘  (disabled — ask a parent)
          if needs_help resolved          → 🔄  (clickable — JIT gets retry_context)
          if seq <= max_mastered + 3      → ▶   (unlocked, clickable)
          if seq == 999                   → ○   (legacy, always unlocked)
          else                            → 🔒  (locked, button disabled)
```

### Dynamic Persona System

| Persona | Age | Avatar | Voice | Assigned when |
|---|---|---|---|---|
| **Pip** | 8 | 👧🏼 | Curious, toy/playground analogies | Simple / concrete facts |
| **Alex** | 13 | 👦🏽 | Slightly sceptical, sports/social analogies | Moderately complex |
| **Riley** | 16 | 🕵️ | Overzealous detective, wild confident-but-wrong theories | Advanced / abstract |

The subheader, chat avatar, placeholder, win message, and error fallbacks all update automatically from `persona_config`. The escalation ladder prompts reference the persona's name but are otherwise persona-agnostic — Pip, Alex, and Riley all escalate through the same three tiers in their own voice.

### Persistence Layer (`student_profile.json`)

```json
{
  "student_name": "Explorer",
  "achievements": {
    "right_angle": {
      "status": "Mastered",
      "frustration_triggers": 1,
      "boss_fight_attempts": 2,
      "was_retry": false
    }
  },
  "needs_help": {
    "photosynthesis": {
      "status": "resolved",
      "attempts": 2,
      "exit_reason": "give_up",
      "timestamp": "2026-05-05T23:09:00",
      "resolved_at": "2026-05-06T08:30:00"
    }
  }
}
```

- `achievements` — keyed by stable slug ID (not display name); written on Phase 2 mastery
- `needs_help` — keyed by same slug ID; `status: active` = locked, `status: resolved` = parent unlocked
- `was_retry: true` in an achievement means the student mastered it after at least one previous abandonment

### Parent Dashboard Sections

| Section | What it does |
|---|---|
| **Summary Metrics** | Concepts Mastered · Learning Friction · 🆘 Needs Help count |
| **Concept Breakdown** | DataFrame with ⭐ Mastered / 🔄 Comeback!, friction triggers, boss-fight attempts |
| **Suggested Home Activity** | Reads `home_activity` from the highest-friction mastered concept |
| **🆘 Needs Help** | Abandoned concepts with exit reason, attempt count, home_activity hint, 🔓 Unlock / 🔒 Re-lock buttons |
| **Review Queue** | ChromaDB `status: pending`; Approve All or individual Approve / Reject & Delete |
| **PDF Upload** | Subject selector + uploader → skeleton extraction → Review Queue |
| **Active Curriculum** | Lists approved concepts; Remove button deletes vector, preserves achievement name |
| **Repair Sequence Numbers** | Assigns proper 1-based sequence numbers to concepts still at `sequence_order=999` |

### Routing — Phase 1 & 2

| Verdict | Phase 1 behaviour | Phase 2 behaviour |
|---|---|---|
| `mastery` | OVERRIDE: validate + present boss scenario | ⭐ WIN: save achievement, balloons |
| `partial_hit` | Validate partial, ask guiding question | Validate partial, ask follow-up |
| `miss` (1st) | Bridging Constraint: anchor to `story_intro` | Nudge toward scenario flaw |
| `miss` (2nd) | Disguised mega-hint as a wondering question | Hint via growing doubt about the scenario |
| `miss` (3+) | Aha! rescue — model the answer in character | "Oh no — MY idea was wrong!" — model the answer |
| `off_topic` | Acknowledge, re-state `story_intro` verbatim | Acknowledge, re-state `verification_scenario` |
| `question` (1–2) | Answer briefly, re-state puzzle verbatim | Answer briefly, re-state scenario verbatim |
| `question` (3+) | Warm exit → `trigger_retry = True` | Warm exit → `trigger_retry = True` |
| `give_up` | OVERRIDE: reveal answer, ask Yes/No retry | OVERRIDE: reveal boss answer, ask Yes/No retry |

### Developer Console

Toggle **🐛 Enable Debug Mode** in the sidebar to reveal:
- **🧠 Background Evaluator** — classification verdict and rationale
- **Persona Generation** — full system prompt sent + raw LLM output
- **💾 Session State** — live JSON dump of all counters, phase, and concept data

---

## Audio Settings (opt-in, off by default)

| Feature | How it works |
|---|---|
| **Speak persona responses (TTS)** | Browser `SpeechSynthesis` API — zero install, Chrome/Edge only. Each persona has a distinct pitch/rate profile. |
| **Mic input — speak your answer (STT)** | `audio_recorder_streamlit` records a WAV clip; `faster-whisper` (base model, local) transcribes it. No cloud call. |

**Mic input flow:**
1. Click the green mic button → turns red when recording
2. Click again (or wait 4 s of silence) to stop
3. Transcribed text appears in an **editable text area** — correct any mishearing
4. Click **Send ✓** to submit, or **Re-record ✗** to discard

**Persona voice profiles:**

| Persona | Pitch | Rate | Effect |
|---|---|---|---|
| Pip (8) | 1.4 | 0.85 | Higher, slower — child-like |
| Alex (13) | 1.1 | 1.0 | Neutral teen voice |
| Riley (16) | 0.9 | 1.1 | Lower, faster — detective energy |

**One-time Whisper model setup (required for mic input):**

```powershell
python scripts/download_whisper_model.py
```

Downloads ~145 MB to `./whisper_model/base/` using plain HTTP (SSL-proxy safe, no `huggingface_hub` dependency). To upgrade model quality:

```powershell
python scripts/download_whisper_model.py --model small   # 460 MB, more accurate
```

Then update `WHISPER_MODEL_PATH = "./whisper_model/small"` at the top of `app.py`.

---

## ingest.py — Batch PDF Ingestion Pipeline

```powershell
# Preview without writing
python ingest.py notes.pdf --dry-run

# Full ingestion: Biology, Module 2
python ingest.py chapter2.pdf --subject Biology --module 2
```

### CLI Options

| Flag | Default | Purpose |
|---|---|---|
| `--dry-run` | off | Print JSON to terminal, do not write to ChromaDB |
| `--model NAME` | `gemma4:e4b` | Ollama model for concept extraction |
| `--embed-model NAME` | `nomic-embed-text` | Ollama embeddings model |
| `--chroma-path PATH` | `./chroma_db` | ChromaDB persistence directory |
| `--subject NAME` | `General` | Subject tag applied to all extracted concepts |
| `--module N` | `1` | Module number for fractional sequencing (`2.1`, `2.2`, …) |
| `--max-chars N` | `3000` | Max characters per semantic chunk |
| `--overlap N` | `300` | Character overlap between adjacent chunks |
| `--chunk-limit N` | all | Process at most N chunks |

### Skeleton Schema (what `ingest.py` stores)

```json
{
  "concept_name": "Photosynthesis Energy Capture",
  "ground_truth_logic": "Plants convert light energy into glucose using CO2 and water...",
  "subject": "Biology",
  "sequence_order": 2.1,
  "status": "pending"
}
```

`story_intro`, `persona_config`, `verification_scenario`, `boss_fight_logic`, and `home_activity` are **not** stored at ingestion time — they are generated by `generate_level_jit()` on every concept click.

---

## ChromaDB Concept Lifecycle

```
ingest.py / in-app PDF upload
        │
        ▼
  status: "pending"    ← Parent Dashboard Review Queue
        │
        ├── ✅ Approve    → status: "approved"  ← visible to students
        │
        └── 🗑️ Reject     → logged to rejected_telemetry.jsonl (DLQ)
                              + physically deleted from ChromaDB
```

---

## Architecture Constraints

- **No agent frameworks** — no LangChain, LangGraph, AutoGen, or OpenAI SDK
- **Raw Python + `requests`** — one dependency for all Ollama HTTP calls
- **Procedural state** — `session_state` keys and plain variables; no state machine libraries
- **Two logical units, one model** — Persona and Evaluator are prompt roles, not separate processes
- **Local vector store** — ChromaDB with `nomic-embed-text` via Ollama; no external DB
- **Privacy-first** — no data leaves the device; all inference and embedding runs on local Ollama

---

## Adding New Concepts

### Via PDF — In-App (recommended)

1. Open **Parent Dashboard** → scroll to "Generate Concept from PDF"
2. Select a subject
3. Upload a PDF — the AI extracts one skeleton per major section
4. Review each concept in the **Review Queue** and click ✅ Approve

### Via PDF — Batch CLI

```powershell
python ingest.py your_material.pdf --dry-run          # preview first
python ingest.py your_material.pdf --subject Math     # write to ChromaDB
```

Then open the Parent Dashboard to review and approve.

---

## What's Next (Planned)

- **Multi-student profiles** — multiple named profiles with a profile switcher in the sidebar
- **Session scoring** — time-to-mastery, miss rate, frustration trend across sessions
- **JIT story caching** — optionally cache the first JIT-generated story for a given student/concept pair so the same narrative persists across sessions while remaining regeneratable on demand
- **Semantic concept search** — use ChromaDB similarity search to surface related concepts when a student struggles, enabling adaptive learning paths
- **Frustration alerts** — Parent Dashboard notification when a concept's friction score crosses a configurable threshold
