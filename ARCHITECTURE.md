# GemmaGenius — System Architecture

A local, privacy-first educational web app built on the **Feynman Technique**.  
The student teaches the AI persona; the AI teaches by asking questions, not by giving answers.  
Everything runs on-device. No cloud APIs. No data leaves the machine.

---

## Core Constraints (Non-Negotiable)

| Constraint | Rationale |
|---|---|
| No agent frameworks (no LangChain, AutoGen, LangGraph) | Zero abstraction — every prompt is auditable Python |
| Raw `requests` / `ollama` client only | Explicit prompt construction; you always know what the model receives |
| Local Ollama models only | Privacy-first; works offline |
| Procedural state (`st.session_state` keys) | No state machine libraries; every transition is readable Python |
| Local ChromaDB vector store | No external DB; embeddings via `nomic-embed-text` through Ollama |

---

## High-Level Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│  app.py  (Streamlit web UI)                                     │
│                                                                 │
│  ┌────────────────┐    ┌───────────────────────────────────┐   │
│  │  Kid Mode       │    │  Parent Dashboard                 │   │
│  │  (Play)         │    │  PDF upload · Review Queue        │   │
│  │  Persona chat   │    │  Needs Help · Active Curriculum   │   │
│  │  Subject sidebar│    │  Analytics · Sequence Repair      │   │
│  └────────────────┘    └───────────────────────────────────┘   │
│                                                                 │
│  generate_level_jit()   build_state_directive()                 │
│  compile_evaluator_prompt()   compile_pip_prompt()              │
│  log_concept_struggle()   resolve_concept_struggle()            │
│  retire_concept_struggle()   save_profile()                     │
└──────────────────┬──────────────────────────────────────────────┘
                   │
       ┌───────────┼──────────────┐
       ▼           ▼              ▼
  Ollama API   ChromaDB       student_profile.json
  (local)      (local)        (local JSON)
```

---

## Two Logical Units, One Ollama Model

Both units hit the same Ollama model (`gemma4:e4b`) but with completely different system prompts:

| Unit | Role | Sees student input? | Output format |
|---|---|---|---|
| **The Evaluator** | Silent judge — grades the student's response against the ground truth | Yes | Strict JSON `{"classification": "..."}` |
| **The Persona** | Confused learner — reacts to the student's explanation in character | Yes (via user prompt) | Natural language |

The Evaluator always runs first. Its verdict determines the `state_directive` injected into the Persona's prompt for that turn. The student never sees the Evaluator.

---

## Turn Lifecycle (per student message)

```
Student submits text (typed or transcribed via Whisper)
        │
        ▼
compile_evaluator_prompt(concept_data, user_input, pips_last_question, phase)
        │
        ▼
call_ollama(evaluator_system, evaluator_user, json_mode=True)
        │  → { "classification": "mastery" | "partial_hit" | "miss" |
        │                         "question" | "off_topic" | "give_up" }
        ▼
build_state_directive(classification, concept_data, phase,
                      frustration_counter, clarification_counter, user_input)
        │  → (directive_string, new_frustration, new_clarification, trigger_retry)
        ▼
compile_pip_prompt(concept_data, directive_string)
        │  → injects persona name/age/voice_tone from concept_data["persona_config"]
        ▼
call_ollama(pip_system, user_input, json_mode=False)
        │
        ▼
append to st.session_state.messages → st.rerun()
```

---

## Evaluator Classification States

| Classification | Meaning |
|---|---|
| `mastery` | Student clearly explained the core mechanism (simple language is fine) |
| `partial_hit` | Relevant ideas present but explanation incomplete or missing the mechanism |
| `miss` | Wrong, confused, guessing, or parroting vocabulary without explaining how it works |
| `question` | Student asked a clarifying question or requested a hint — ALWAYS `question`, never `give_up` |
| `off_topic` | Joke, nonsense, or completely unrelated topic |
| `give_up` | Unambiguous explicit surrender ("I give up", "just tell me") — a wrong answer is still `miss` |

---

## Frustration Escalation Ladder

`frustration_counter` is incremented **before** the directive is chosen (post-increment thresholds):

### Phase 1 — Elicitation misses

| `new_frustration` | Persona behaviour |
|---|---|
| **1** (first miss) | **Bridging Constraint** — acknowledge user's attempt, bridge back to `story_intro`, ask again. No hints. |
| **2** (second miss) | **Disguised mega-hint** — persona voices growing doubt, poses a question that points almost directly at the core truth without stating it |
| **≥ 3** (rescue) | **Aha! moment** — persona "suddenly realises" the answer, models the correct explanation in character, asks student to echo it back |

### Phase 2 — Boss Fight misses (mirrors Phase 1)

| `new_frustration` | Persona behaviour |
|---|---|
| **1** | Act confused, nudge toward the flaw without revealing it |
| **2** | Voice growing doubt about the scenario, embed a hint as a wondering question |
| **≥ 3** | "Oh no — wait. I think MY idea was wrong the whole time!" — models the correct answer, asks student to explain why |

The `clarification_counter` escalates independently:

| `clarification_counter` | Behaviour |
|---|---|
| **< 2** | Answer the question briefly in-character, re-state the puzzle verbatim |
| **≥ 2** | Warm exit: "let's take a break and ask a grown-up" → `trigger_retry = True` |

`trigger_retry = True` is the fourth return value from `build_state_directive()`. The call-site sets `awaiting_retry = True`, which surfaces the Yes/No retry UI (same as `give_up`) — the student is never left trapped in a silent chat.

---

## Session State Variables

Initialised by `start_session()` at the start of every concept:

| Key | Type | Purpose |
|---|---|---|
| `current_phase` | `int` (1 or 2) | Elicitation vs. Boss Fight |
| `frustration_counter` | `int` | Miss count — drives escalation ladder |
| `clarification_counter` | `int` | Question count — drives clarification escalation |
| `pips_last_question` | `str` | Seeded into the Evaluator each turn for context |
| `awaiting_retry` | `bool` | True when Yes/No retry UI should replace chat input |
| `session_won` | `bool` | True after Phase 2 mastery — shows balloon + win message |
| `session_ended` | `bool` | True after graceful exit — shows "choose a new concept" |
| `balloons_shown` | `bool` | One-shot guard — prevents balloon from refiring on micro-reruns |
| `concept_data` | `dict` | Full concept merged from ChromaDB skeleton + JIT output |
| `current_concept_id` | `str` | Stable slug ID used as the key in `student_profile.json` |
| `messages` | `list` | Full chat history (role, content) |
| `last_spoken_idx` | `int` | TTS pointer — prevents re-speaking messages on rerun |
| `pending_transcription` | `str` | Holds transcribed mic text awaiting user review/edit |
| `last_audio_hash` | `str` | MD5 of last submitted audio — prevents double-submission |

---

## Two-Phase Session Structure

### Phase 1 — Elicitation
Persona presents `story_intro`. Evaluator grades against `ground_truth_logic` (ChromaDB) or `evaluator_ground_truth` (legacy). Both field names are supported via fallback chains.

- `mastery` → seamless transition to Phase 2 in a single Persona turn (no extra Ollama call for the transition text)
- All other verdicts → escalation ladder (see above)

### Phase 2 — Boss Fight (Verification)
Persona presents `verification_scenario` — a scenario containing a deliberate logical mistake. Evaluator grades against `boss_fight_logic` (ChromaDB) or `verification_ground_truth` (legacy).

- `mastery` → TRUE WIN STATE: save achievement, fire balloons, surface win message
- All other verdicts → Phase 2 escalation ladder

---

## JIT Architecture (Just-In-Time Level Generation)

`ingest.py` stores only a **skeleton** in ChromaDB — `concept_name` and `ground_truth_logic`. All story content is generated on demand when the student clicks a concept.

```
INGESTION (ingest.py)              PLAY TIME (app.py)
─────────────────────              ──────────────────
PDF text
  → semantic chunks with overlap
  → Ollama extracts skeleton:
      concept_name
      ground_truth_logic         Student clicks concept in sidebar
  → nomic-embed-text embedding            │
  → ChromaDB upsert                       ▼
      status: "pending"          generate_level_jit(
                                   concept_name,
Parent approves in Dashboard       ground_truth_logic,
  → status: "approved"             skeleton = concept_obj,
                                   retry_context = nh_entry  ← if resolved struggle
                                 )
                                          │
                                          │  Ollama generates:
                                          │    persona_config (Pip/Alex/Riley)
                                          │    story_intro (random setting seed)
                                          │    verification_scenario
                                          │    boss_fight_logic
                                          │    home_activity
                                          ▼
                                 start_session(merged_concept)
```

**Narrative Continuity:** when `retry_context` is supplied (the concept has a `needs_help` entry with `status: "resolved"`), the JIT prompt includes a **Narrative Continuity Directive** instructing the persona to open by warmly acknowledging the previous struggle rather than starting fresh.

---

## Student Struggle State Machine

Tracks non-mastery sessions in `student_profile.json["needs_help"]`:

```
First attempt ends without mastery
        │
        ▼  log_concept_struggle(cid, exit_reason)
needs_help[cid] = { status: "active", attempts: 1, exit_reason, timestamp }
        │
        │  Sidebar: 🆘 (disabled button)
        │
        ▼  Parent opens Dashboard → sees Needs Help section → does home_activity IRL
           Parent clicks 🔓 Unlock
        │
        ▼  resolve_concept_struggle(cid)
needs_help[cid].status = "resolved"   (attempt history preserved)
        │
        │  Sidebar: 🔄 (enabled button)
        │
   ┌────┴────────────────────────────────────────────────────┐
   │  Child clicks 🔄                                        │
   │    generate_level_jit(retry_context = needs_help[cid])  │
   │    Persona opens with narrative continuity              │
   └──────────────┬──────────────────────────────────────────┘
                  │
      ┌───────────┴──────────────────┐
      │ MASTERY                      │ FAILURE AGAIN
      ▼                              ▼
retire_concept_struggle(cid)     log_concept_struggle(cid)
achievements[cid] = {            → status back to "active"
  status: "Mastered",            → attempts incremented
  was_retry: True,               → parent must intervene again
  ...
}
Sidebar: ⭐
```

Three helper functions manage all transitions:

| Function | Transition | When called |
|---|---|---|
| `log_concept_struggle(cid, reason)` | → `active` (or re-locks `resolved`) | On any non-mastery exit |
| `resolve_concept_struggle(cid)` | `active` → `resolved` | Parent clicks 🔓 Unlock |
| `retire_concept_struggle(cid)` | `resolved` → deleted | Phase 2 mastery win |

---

## Concept Lifecycle (ChromaDB)

```
ingest.py / in-app PDF upload
        │
        ▼
  status: "pending"    ← Parent Dashboard Review Queue
        │
        ├── ✅ Approve    → status: "approved"  ← visible to students in Kid Mode
        │
        └── 🗑️ Reject     → logged to rejected_telemetry.jsonl (DLQ)
                             + physically deleted from ChromaDB vector store
```

---

## Sidebar Locking Logic

```
Pass 1: find max_mastered_seq
  — highest sequence_order among concepts where cid ∈ achievements

Pass 2: for each concept in subject folder (sorted by sequence_order):

  is_mastered   = cid ∈ achievements
  nh_entry      = needs_help_map.get(cid, {})
  nh_status     = nh_entry.get("status", "")    # "active" | "resolved" | ""
  is_needs_help = nh_entry is non-empty AND not mastered

  is_unlocked = is_mastered
             OR seq == 999                      (legacy / pre-sequencing)
             OR seq <= max_mastered_seq + 3     (look-ahead buffer)

  if is_unlocked:
    if is_mastered             → ⭐  (clickable)
    if is_needs_help + active  → 🆘  (disabled, tooltip: ask a parent)
    if is_needs_help + resolved→ 🔄  (clickable, JIT gets retry_context)
    if is_playing              → ▶   (clickable)
    else                       → ○   (clickable)
  else:
                               → 🔒  (disabled)
```

---

## Dynamic Persona System

Persona is stored in `concept_data["persona_config"]` and drives the entire Kid Mode UI:

| Persona | Age | Avatar | Voice | Assigned when |
|---|---|---|---|---|
| **Pip** | 8 | 👧🏼 | Curious, toy/playground analogies | Simple / concrete facts |
| **Alex** | 13 | 👦🏽 | Slightly sceptical, sports/social analogies | Moderately complex |
| **Riley** | 16 | 🕵️ | Overzealous detective, invents wild theories | Advanced / abstract |

The subheader, chat avatar, chat placeholder, win message, and error fallbacks all update automatically from `persona_config`. The escalation ladder prompts reference the persona's name but are otherwise persona-agnostic.

---

## Audio Layer

Both audio features are opt-in (off by default):

| Feature | Implementation | Key detail |
|---|---|---|
| **TTS** (speak responses) | Browser `SpeechSynthesis` API via `st.components.v1.html()` | Zero Python dependencies; Chrome/Edge only; per-persona pitch/rate profiles |
| **STT** (mic input) | `audio_recorder_streamlit` widget + local `faster-whisper` (base model) | Audio deduped via MD5 hash; transcription staged in `pending_transcription` for user review/edit before submission |

**Mic input flow:**
1. Student records clip → `audio_recorder_streamlit` returns bytes
2. MD5 hash checked against `last_audio_hash` — prevents double-submission on Streamlit rerun
3. `transcribe_audio()` calls local Whisper model, returns text string
4. Text staged in `st.session_state.pending_transcription`
5. Editable `st.text_area` rendered with "Send ✓" / "Re-record ✗" buttons
6. Student confirms → dispatched into the normal turn pipeline

---

## Persistence Files

| File | Format | Purpose |
|---|---|---|
| `student_profile.json` | JSON | `achievements` (mastery records) + `needs_help` (struggle records) |
| `chroma_db/` | SQLite (binary) | Local ChromaDB vector store — gitignored |
| `rejected_telemetry.jsonl` | JSONL | Dead Letter Queue — append-only rejected concept log |
| `knowledge.json` | JSON | Legacy concept graph — `main.py` only; fallback for `app.py` |
| `whisper_model/base/` | Binary | Local faster-whisper model bundle — gitignored |

---

## Parent Dashboard Sections

| Section | What it does |
|---|---|
| **Summary Metrics** | Concepts Mastered · Learning Friction · 🆘 Needs Help count |
| **Concept Breakdown** | DataFrame with per-concept status (⭐ Mastered / 🔄 Comeback!), friction, boss-fight attempts |
| **Suggested Home Activity** | Reads `home_activity` from highest-friction mastered concept |
| **🆘 Needs Help** | Lists abandoned concepts with exit reason, attempt count, home_activity hint, 🔓 Unlock / 🔒 Re-lock buttons |
| **Review Queue** | ChromaDB `status: pending` concepts; Approve All or individual Approve/Reject |
| **PDF Upload** | Subject selector + uploader → JIT skeleton extraction → Review Queue |
| **Active Curriculum** | Lists all approved concepts; Remove button deletes vector, preserves achievement name |
| **Repair Sequence Numbers** | Assigns proper 1-based sequence numbers to concepts still at `sequence_order=999` |

---

## ingest.py — Batch Pipeline

```
PDF
  → pypdf text extraction
  → semantic_chunk_text(chunk_size=3000, overlap=300)
      splits on \n\n (paragraphs) first, sentence fallback
      carries 300-char tail of previous chunk into next (overlap)
  → for each chunk:
      Ollama (gemma4:e4b) extracts skeleton JSON
        { concept_name, ground_truth_logic }
      slugify(concept_name) → deterministic ChromaDB ID
      ollama.embeddings(nomic-embed-text, searchable_text) → vector
      collection.upsert(id, vector, document, metadata)
        metadata includes: subject, sequence_order, status="pending"
```

CLI flags: `--dry-run`, `--model`, `--embed-model`, `--subject`, `--module`, `--max-chars`, `--overlap`, `--chunk-limit`.

---

## What Is NOT Built (Planned)

- **Multi-student profiles** — multiple named profiles in `student_profile.json` with a profile switcher
- **Session scoring** — time-to-mastery, miss rate, frustration trend across sessions
- **JIT story caching** — optional: persist the first JIT-generated story so a student sees the same narrative on return visits (currently regenerates on every click)
- **Semantic concept search** — use ChromaDB similarity search to surface related concepts when a student struggles
- **Frustration alerts** — push notification / badge on Parent Dashboard when a concept's friction score crosses a configurable threshold
