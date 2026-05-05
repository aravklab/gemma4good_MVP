# GemmaGenius — Project Learnings

A living document of hard-won lessons from building a local, AI-powered educational app with Streamlit, Ollama, and ChromaDB. Written to inform future development and serve as a reference for anyone building similar systems.

---

## 1. LLM Prompt Engineering

### 1.1 Single-concept prompts silently discard data
The original prompt told the LLM to "extract ONE key concept." On a 10-section PDF it would extract one good concept and stop — no error, no warning, just silence. The fix was to reframe the instruction: *"Extract a distinct concept for EVERY major section or numbered heading."* Explicit cardinality in the prompt matters.

### 1.2 "Early stopping" is a structural prompt problem, not a model bug
When the LLM stopped after one concept we initially assumed it was a model limitation. It was actually the prompt. The old prompt had a schema that implied a single JSON object at the root. Wrapping the output in `{ "extracted_concepts": [...] }` gave the model a container and it filled it.

### 1.3 Output format instructions need to be unambiguous
Instructions like "Output raw JSON only" are not enough. The model will still wrap responses in markdown fences (` ```json ``` `) occasionally. Defensive parsing — stripping fences before `json.loads()` — is always necessary. A `{"skip": true}` escape hatch for low-quality input is also valuable.

### 1.4 The `{"skip": true}` pattern is a clean escape valve
Rather than forcing the LLM to generate a concept from meaningless text (e.g., a table of contents page), allowing it to return `{"skip": true}` and handling that gracefully in code produces far fewer hallucinations than insisting it always output something.

### 1.5 Persona voice must be injected into the system prompt, not assumed
Early versions assumed the LLM would "know" Pip was 8 years old from context. The persona drifted toward generic assistant behaviour. Explicitly injecting `{persona['name']}, {persona['age']}, {persona['voice_tone']}` into the system prompt string locked the voice in reliably.

### 1.6 The "Bridging Constraint" eliminates topic drift on wrong answers
Without it, when a student gave a wrong answer the persona would freestyle — inventing new analogies, drifting to unrelated topics. Forcing the prompt to include the exact original `story_intro` and the student's last message made the persona anchor back to the puzzle every time. Verbatim injection of source context > vague instructions to "stay on topic."

### 1.7 Two-agent architecture prevents the persona from self-correcting
Separating the Evaluator (silent judge, grades the student) from the Persona (actor, never reveals the answer) was the single most important architectural decision. A single agent in both roles will leak the answer — the model's "helpfulness" bias is too strong to override with instructions alone.

---

## 2. ChromaDB and Vector Storage

### 2.1 ChromaDB metadata must be flat scalars
ChromaDB does not accept nested dicts or lists in the `metadatas` field. The workaround: serialize the full concept JSON to a string (`json.dumps(concept)`) and store it as a single metadata field (`concept_json`). Surface only the fields needed for querying (e.g. `status`, `concept_name`) as top-level metadata keys.

### 2.2 `upsert` over `add` — always
Using `.add()` will throw an error if an ID already exists, which breaks re-ingestion of updated PDFs. `.upsert()` is idempotent and should be the default for any content pipeline.

### 2.3 Deterministic IDs beat random UUIDs for educational content
Random UUIDs mean re-ingesting the same PDF creates duplicate entries. Slugifying the concept name into an ID (e.g. `"Right Angle"` → `"right_angle"`) makes upsert idempotent and makes the database inspectable without a query tool.

### 2.4 `status` metadata field enables a simple review workflow
Storing `status: "pending"` on every ingested concept and changing it to `"approved"` on parent approval is a lightweight content moderation layer that requires zero extra infrastructure. The query `collection.get(where={"status": "pending"})` is the entire queue implementation.

### 2.5 Physical deletion is the right call for rejected concepts
Keeping rejected concepts as `status: "rejected"` in the vector store pollutes future semantic searches — similar-but-bad content will surface as neighbours to good content. Deleting the vector and logging to an append-only JSONL file (Dead Letter Queue) gives you auditability without contamination.

### 2.6 `@st.cache_resource` is essential for the ChromaDB client
Initialising a `chromadb.PersistentClient` on every Streamlit re-render causes file-lock conflicts and is slow. Wrapping it in `@st.cache_resource` creates exactly one client per server process.

---

## 3. Python Environment Management

### 3.1 Multiple Python installations silently cause `ModuleNotFoundError`
The machine had Python 3.12, Python 3.14, and a `.venv` virtual environment. Installing `chromadb` with `pip install` went to the wrong interpreter. The symptom was a clean `ModuleNotFoundError` in Streamlit despite the package appearing installed. Fix: always install using the *exact* interpreter that runs the app — `& "path\to\.venv\Scripts\python.exe" -m pip install package`.

### 3.2 Find the actual interpreter before installing anything new
```powershell
Get-ChildItem -Recurse -Filter "streamlit.exe" | Select FullName
```
This reveals which virtual environment owns the running Streamlit process. Install all new packages there.

### 3.3 `sys.stdout.reconfigure(encoding="utf-8")` is a Windows necessity
PowerShell's default encoding is `cp1252`. Any emoji or Unicode character in print output (persona avatars, bullet points) will raise a `UnicodeEncodeError`. Adding this at the top of CLI scripts eliminates the problem:
```python
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```

---

## 4. Streamlit-Specific Lessons

### 4.1 Switching views must not reset game state
Storing the active chat in `st.session_state.messages` (not re-initialising it on view change) means a parent can check the dashboard and the kid's game is exactly where they left it. Never key session state to the active nav view.

### 4.2 `st.stop()` is the right way to halt mid-render
After showing an error or warning in the middle of a Streamlit callback, call `st.stop()` to prevent the rest of the function from executing. Using `return` alone doesn't always work as expected inside nested callbacks.

### 4.3 `st.rerun()` must be the last call in any state-mutating block
Any code after `st.rerun()` will not execute. Treat it like `return` — it immediately rerenders the page.

### 4.4 The fail-fast guardrail (500 char minimum) prevents wasted LLM calls
Scanned or image-heavy PDFs extract as nearly empty text. Checking `len(extracted_text) < 500` before any Ollama call saves time and produces a much cleaner user error message than a malformed JSON response from the model.

### 4.5 Debug mode as a sidebar toggle is worth the time investment
Adding a `🐛 Enable Debug Mode` checkbox that exposes the raw evaluator output, raw persona prompt, and session state JSON cut debugging time dramatically. The LLM's classification rationale is especially useful — it explains *why* a student answer was marked `miss` vs `partial_hit`.

---

## 5. Semantic Chunking

### 5.1 Character-based chunking cuts concepts in half
Splitting a PDF at every 3,000 characters frequently splits a concept mid-sentence, mid-equation, or mid-example. The LLM then receives incomplete context and generates nonsense or fills the gap with hallucinations.

### 5.2 Paragraph-first splitting with overlap is the right default
Splitting on `\n\n` (paragraph breaks) first, then falling back to sentence boundaries, keeps conceptual units intact. Adding a 300-character overlap (carry the end of the previous chunk into the start of the next) ensures that concepts near a boundary appear in full in at least one chunk.

### 5.3 Large single paragraphs need a hard safety valve
Some PDFs (especially academic papers) have paragraphs that are thousands of characters long. The chunker needs a force-slice fallback for paragraphs that exceed the chunk size limit, otherwise the entire PDF can end up as one chunk.

---

## 6. Content Pipeline Design

### 6.1 Separation of ingestion and play is critical for quality control
Early versions wrote directly to `knowledge.json` during ingestion. This meant LLM hallucinations went straight into gameplay. The `pending → approved` pipeline introduced a human review gate that catches bad concepts before any student sees them.

### 6.2 The Dead Letter Queue (DLQ) turns rejections into training data
`rejected_telemetry.jsonl` stores every rejected concept with its rejection reason. Over time this becomes a dataset that shows which prompt patterns produce hallucinations, which source PDFs produce bad output, and which rejection categories are most common — valuable signal for future prompt iteration.

### 6.3 Home activities should be generated, not hardcoded
The original code had a `CONCEPT_ACTIVITIES` dictionary with hardcoded activity suggestions per concept. It became stale immediately and required code changes for every new concept. Moving `home_activity` into the LLM schema means every concept arrives with its own tailored activity suggestion — no maintenance required.

---

## 7. Git and Collaboration

### 7.1 PowerShell does not support Bash heredoc syntax
Multi-line git commit messages using `$(cat <<'EOF' ... EOF)` fail in PowerShell. Options: write the message to a temp file and use `git commit -F`, or keep messages single-line. The temp-file approach is reliable.

### 7.2 `chroma_db/` must be in `.gitignore`
The ChromaDB directory contains binary SQLite files that are large, machine-specific, and change on every write. Committing it pollutes history and causes merge conflicts. Every developer regenerates it locally from the ingestion pipeline.

### 7.3 `rejected_telemetry.jsonl` should be committed as a stub
Unlike `chroma_db/`, the DLQ log is valuable institutional memory (rejected concepts, rejection reasons, timestamps). Committing an empty stub file means the file path is established in the repo and the append-only format starts from a known state.

---

## 8. Architectural Decisions Worth Keeping

| Decision | Why it holds |
|---|---|
| No agent frameworks (no LangChain etc.) | Zero abstraction layers means every prompt is auditable and every state transition is readable Python |
| Flat `requests` calls to Ollama | Forces explicit prompt construction — you always know exactly what the model is receiving |
| `student_profile.json` on disk | Dead-simple persistence with no database dependency; trivially human-readable for debugging |
| JSONL for the DLQ | Append-only, streamable, compatible with `jq` and `pandas` for analysis |
| Stable slugified IDs in ChromaDB | Idempotent re-ingestion; predictable IDs for debugging; no orphan duplicates |
| Two logical LLM roles in one model | Avoids running two Ollama processes; the prompt separation is sufficient for the use case |

---

## 9. Behavioural State Machines in Educational Apps

### 9.1 Tracking without acting is useless telemetry
`frustration_counter` was incremented on every miss from day one, but the LLM received the same directive regardless of whether it was miss #1 or miss #10. The counter was measuring but never changing behaviour. The fix — branching on post-increment thresholds inside `build_state_directive()` — is simple, but the lesson is: any metric you track should eventually influence something, or it's just noise.

### 9.2 Post-increment vs pre-increment matters for threshold logic
When `new_frustration += 1` runs first and thresholds are checked on `new_frustration`, tier 1 fires at 1, tier 2 at 2, tier 3 at ≥ 3. If you check `frustration_counter` before incrementing, every tier fires one attempt too early. This is a subtle off-by-one that looks correct in isolation but produces confusing UX.

### 9.3 Dead-end states that don't exit are silent bugs
The clarification counter had a threshold ("let's look at a book together") that produced a message but no exit path. The chat input remained open and the AI would repeat the same message indefinitely. Any escalation path that changes a persona's dialogue MUST be paired with an exit path — a retry prompt, a session-end flag, or a return to the concept selector.

### 9.4 Returning a fourth boolean from routing functions is cleaner than side effects
`build_state_directive()` originally returned `(directive, new_frustration, new_clarification)`. Adding a `trigger_retry` boolean as the fourth return value let the call-site react to the dead-end without `build_state_directive()` needing to know about Streamlit session state. The function stays pure (Python logic only); all UI effects happen at the call-site.

### 9.5 Hard-delete on unlock loses pedagogically valuable history
When a parent clicks "Unlock" on a concept the child abandoned, deleting the struggle record gives the AI total amnesia. The persona opens as if the previous session never happened, which breaks narrative immersion. The solution: a three-state status field (`active` / `resolved` / deleted-on-mastery) that preserves attempt history and allows the JIT prompt to inject a Narrative Continuity Directive on the retry session.

### 9.6 "Comeback" wins deserve their own achievement badge
`was_retry: True` in the achievements record lets the Parent Dashboard distinguish a first-attempt mastery from a mastery achieved after parent intervention and multiple failures. This is motivationally important — it should be celebrated differently, not silently merged with first-attempt wins.

---

## 10. Audio Integration

### 10.1 `huggingface_hub` is incompatible with corporate SSL-inspection proxies
`faster-whisper`'s default download path uses `huggingface_hub.snapshot_download()`, which internally uses `hf-xet` (a Rust-based transfer utility). Corporate proxies that perform SSL inspection replace certificates, causing `CERTIFICATE_VERIFY_FAILED`. Setting `REQUESTS_CA_BUNDLE=""` or `huggingface_hub.configure_http_backend` didn't help because `hf-xet` bypasses Python's HTTP stack entirely. The only reliable fix: bypass `huggingface_hub` completely and download model files directly with Python's `urllib.request` and `ssl._create_unverified_context`.

### 10.2 `faster-whisper` expects `vocabulary.txt`, not `vocabulary.json`
The vocabulary file required by `faster-whisper` is named `vocabulary.txt`. The initial download script listed `vocabulary.json`, which silently 404'd (the script treated 404 as an optional file). The model loaded but failed with "Cannot load the vocabulary from the model directory". Always verify exact filenames against the model repository before writing a custom download script.

### 10.3 `@st.cache_resource` caches `None` from failed loads
If `_load_whisper()` fails and returns `None`, `@st.cache_resource` caches that `None`. Subsequent calls return `None` without re-attempting the load — even after the underlying problem (missing files) is fixed. A full Streamlit server restart is required to clear the cache. Add a defensive `if model is None: return None` check and document this behaviour clearly.

### 10.4 `audio_recorder_streamlit` holds audio bytes across reruns
The widget retains the last recorded audio bytes in its component state across Streamlit reruns. Without a deduplication guard, every rerun (including the rerun triggered by `st.rerun()` after displaying the transcription) re-submits the same audio clip. Fix: compute `hashlib.md5(audio_bytes).hexdigest()` and compare against `st.session_state.last_audio_hash`. Only process the clip if the hash is new.

### 10.5 Never call `st.balloons()` immediately before `st.rerun()`
`st.rerun()` aborts the current script execution immediately. Any `st.balloons()` call before it is discarded before reaching the browser. The correct pattern: set a `balloons_shown` flag in session state, check it in the win-render block (after `render_chat_history()` completes), fire once, and set the flag. This also prevents the balloon from re-firing on every subsequent micro-rerun.

### 10.6 CSS `box-shadow` on cross-origin iframes produces misaligned animations
Applying a `box-shadow` animation to an `audio_recorder_streamlit` iframe via a CSS selector targets the full iframe element, which is wider than the circular button inside it. The shadow renders as an elongated oval, not a ring. Since the component is cross-origin, JavaScript cannot inspect or style its internals. The fix: inject a separate animated HTML element (`<span>` with CSS keyframes) directly inside the mic column via `st.markdown(unsafe_allow_html=True)`. This is entirely self-contained and layout-safe.

### 10.7 Transcription staging improves accuracy and reduces frustration
Dispatching transcribed text directly into the AI pipeline (common pattern) means mis-heard words go straight to the LLM without any chance to correct them. Staging the transcription in `pending_transcription` and showing an editable `st.text_area` with "Send ✓" / "Re-record ✗" buttons adds one interaction step but dramatically reduces frustration from Whisper errors on accents, background noise, or technical vocabulary.

---

## 11. Things We Would Do Differently

- **Start with ChromaDB from day one.** Migrating from `knowledge.json` to ChromaDB mid-project required rewriting both the ingestion pipeline and the review queue UI. Designing around a vector store from the start would have been cleaner.
- **Design the review queue before the ingestion pipeline.** The first version of `ingest.py` wrote directly to disk with no review step. Retroactively adding a `pending` gate required refactoring both ends.
- **Test with bad PDFs early.** Scanned images, copy-protected files, and tables-only documents all produce edge-case failures that reveal weak spots in the pipeline. Including them in early testing would have surfaced the fail-fast guardrail need sooner.
- **Pin the virtual environment path in a `Makefile` or `justfile`.** Managing `pip install` across multiple Python installations on Windows caused repeated `ModuleNotFoundError` issues. A single `make install` target pointing at the correct interpreter would eliminate the problem.
