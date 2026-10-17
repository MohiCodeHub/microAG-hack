# Live Annotation & Fraud Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A phone-in-browser recording session that is captioned every 10 s by a stateless VLM, watched by a stateful text-only monitor that compares it to reference timelines, asks the worker short spoken questions on divergence, and ends with a step segmentation and fraud score.

**Architecture:** FastAPI server on the laptop (HTTPS) holds all keys. The phone page streams 1 FPS JPEG frames + IMU over a WebSocket; the server batches 10 frames into a chunk, captions it with a fresh `gpt-5.6-luna` call, runs cheap signals, then feeds the text timeline to the monitor. Agent speech goes back to the phone (audio) and the dashboard (text). Dashboard on the laptop shows everything live, and can also act as the source by playing an MP4 in real time.

**Tech Stack:** Python 3.14, FastAPI + uvicorn, `openai` (AsyncOpenAI, Responses API, strict JSON schema), `httpx` (ElevenLabs), Pillow, `imagehash`, numpy, PyAV (reference builder), pytest + pytest-asyncio. Frontend: vanilla HTML/JS/CSS, no build step. `uv` for the venv.

**Spec:** `docs/superpowers/specs/2026-10-17-live-annotation-agent-design.md`

## Global Constraints

- Captioner model `gpt-5.6-luna`; monitor model `gpt-5.6-luna` by default, switchable to `gpt-6-astra` via `MONITOR_MODEL` env var.
- Chunk = 10 frames at 1 FPS = 10 s. Images go **only** to the captioner, never to the monitor.
- Captioner output field order: `caption, step, confidence, objects, hands_visible (both|one|none), activity (active|idle), recapture_suspected (bool)`.
- Captioner candidate steps = claimed task's steps + steps of up to 2 other tasks in the library (fallback distractors `idle_or_waiting`, `using_phone`, `unrelated_tidying` when the library has no other task) + `other`.
- Monitor output: `current_step, progress (0–1), status (ok|watch|diverge), divergence_type (none|off_script|wrong_task|out_of_order|padding|idle|recapture), reason, wake (bool), question (≤12 words), fraud_risk (0–1), explained (true|false|null)`.
- Question cooldown 45 s after any question; at most 1 question per 60 s; never while a question is pending. Hands nudge text "Keep your hands in view, please." with its own 20 s cooldown, no monitor involvement.
- Answer timeout 20 s → event marked unexplained.
- Monitor prompt order: instructions, reference timelines, then session timeline (append-only), then newest chunk — stable prefix for prompt caching.
- Timeline line formats: `mm:ss–mm:ss caption [step]` (chunk), `mm:ss AGENT: "..."`, `mm:ss WORKER: "..."`.
- Keys only from `.env` (`OPENAI_API_KEY`, `ELEVENLABS_API_KEY`); never sent to the browser; `.env` is gitignored.
- Task folder format: `tasks/<task_id>/task.json` = `{"id", "name", "steps": [{"id", "label"}]}`, reference videos `tasks/<task_id>/refs/*.mp4`, built timelines `tasks/<task_id>/refs/*.timeline.json`.
- Recording continues during questions; no pausing.
- Single active session at a time (demo tool).

## Review Focus

1. **LLM call fails, times out (>20 s) or returns unparseable JSON** → chunk is recorded with caption `"(caption unavailable)"` / monitor decision defaults to `status=ok, wake=false`; the session never crashes. (Tests in Task 4 and Task 6.)
2. **Monitor slower than 10 s** → chunks queue and are monitored strictly in order; no caption is dropped or reordered. (Test in Task 6.)
3. **Worker says nothing or STT returns empty text** → recorded as `WORKER: "(no answer)"`, event unexplained, no re-ask loop. (Test in Task 6.)
4. **Task with no reference timelines** → monitor runs with the step list only and its prompt says no references exist; `build_refs` on a task with no MP4s exits with a clear message. (Tests in Task 1 and Task 4.)
5. **iOS Safari blocks audio autoplay and DeviceMotion until a tap** → the phone's Start button unlocks audio and requests motion permission in the same gesture; if motion is denied, the session still runs without IMU. (Manual check in Task 8.)

---

## File Structure

```
app/
  config.py      settings from env + tuning constants
  tasks.py       task library, reference timelines, candidate step lists
  timeline.py    Chunk / Timeline, text rendering
  policy.py      wake + nudge cooldown policies
  signals.py     perceptual-hash duplicate store, visual motion, IMU energy, mismatch
  llm.py         captioner, monitor, final report (OpenAI)
  voice.py       ElevenLabs TTS + STT
  report.py      deterministic signal stats + final score
  session.py     orchestrator: batching, pipelining, wake/answer flow, finish
  server.py      FastAPI app, WebSockets, static files
scripts/
  build_refs.py  MP4s → reference timelines
  make_cert.sh   self-signed cert for the laptop LAN IP
  run.sh         start uvicorn over HTTPS
static/
  capture.js     shared: FrameSampler, ImuBuffer, VoicePlayer, AnswerRecorder, Link (WS)
  phone.html / phone.js
  dashboard.html / dashboard.js
  style.css
tasks/example_make_tea/task.json   placeholder task until teammates drop real ones in
tests/
```

---

### Task 1: Project scaffold, config, task library

**Files:**
- Create: `pyproject.toml`, `app/__init__.py`, `app/config.py`, `app/tasks.py`, `tasks/example_make_tea/task.json`, `tests/test_tasks.py`, `.env.example`

**Interfaces:**
- Produces:
  - `config.Settings` (pydantic-free dataclass) loaded by `get_settings()`: `openai_api_key, elevenlabs_api_key, caption_model="gpt-5.6-luna", monitor_model (env MONITOR_MODEL, default "gpt-5.6-luna"), elevenlabs_voice_id, tasks_dir=Path("tasks"), data_dir=Path("data"), chunk_frames=10, llm_timeout_s=20, question_cooldown_s=45, question_min_interval_s=60, nudge_cooldown_s=20, answer_timeout_s=20`.
  - `tasks.Task(id: str, name: str, steps: list[Step])`, `tasks.Step(id: str, label: str)`
  - `tasks.load_library(tasks_dir: Path) -> dict[str, Task]`
  - `tasks.candidate_steps(task: Task, library: dict[str, Task]) -> list[Step]` — task steps, then up to 2 other tasks' steps (sorted by task id for determinism), else fallback distractors, then `Step("other","something else")`.
  - `tasks.load_reference_timelines(tasks_dir: Path, task_id: str) -> list[str]` — rendered timeline text per `*.timeline.json`, `[]` if none.

- [ ] **Step 1:** `uv init --no-workspace` style `pyproject.toml`; deps `fastapi uvicorn[standard] openai httpx python-dotenv pillow imagehash numpy av`; dev deps `pytest pytest-asyncio`. `uv sync`. If `av` has no wheel for 3.14, pin the venv to `uv python install 3.13` and note it in the commit.
- [ ] **Step 2: Write failing tests** in `tests/test_tasks.py`:
  - `test_load_library_reads_task_json` — tmp dir with two tasks → keys equal both ids, steps parsed in order.
  - `test_candidate_steps_includes_distractors_and_other` — library {a, b, c}; for `a`: first steps are a's, contains b's and c's step ids, last id is `"other"`.
  - `test_candidate_steps_fallback_when_alone` — library {a}: contains `idle_or_waiting`, `using_phone`, `unrelated_tidying`, ends with `other`.
  - `test_reference_timelines_empty_when_no_refs` → `[]`.
  - `test_reference_timelines_renders_lines` — one `x.timeline.json` with 2 chunks → string contains `"00:00–00:10"` and `"00:10–00:20"`.
- [ ] **Step 3:** Run `uv run pytest tests/test_tasks.py -v` → FAIL (import errors).
- [ ] **Step 4:** Implement `config.py`, `tasks.py`; reference timeline JSON is the `Timeline.to_json()` shape from Task 2 — implement rendering via `Timeline.from_json(...).render()` (do Task 2's `timeline.py` first if executing out of order). Add `tasks/example_make_tea/task.json` with 6 steps (fill kettle, boil, cup + teabag, pour, steep/remove bag, add milk).
- [ ] **Step 5:** Run tests → PASS. Commit `feat: scaffold, config, task library`.

### Task 2: Timeline and policies

**Files:** Create `app/timeline.py`, `app/policy.py`, `tests/test_timeline.py`, `tests/test_policy.py`

**Interfaces:**
- Produces:
  - `timeline.Chunk(index: int, t_start: float, t_end: float, caption: str, step: str, confidence: float, objects: list[str], hands_visible: str, activity: str, recapture_suspected: bool)`
  - `timeline.Timeline`: `add_chunk(c: Chunk)`, `add_agent(t: float, text: str)`, `add_worker(t: float, text: str)`, `render() -> str`, `chunks: list[Chunk]`, `to_json() -> dict`, `from_json(d) -> Timeline`
  - `timeline.fmt(t: float) -> str` → `"mm:ss"`
  - `policy.WakePolicy(cooldown_s, min_interval_s)`: `can_ask(now: float, pending: bool) -> bool`, `record_question(now: float)`
  - `policy.NudgePolicy(cooldown_s)`: `should_nudge(now: float, hands_visible: str) -> bool` (records when it returns True)

- [ ] **Step 1: Failing tests:**
  - `test_render_orders_lines_by_time` — chunk 0–10, agent at 12, worker at 15, chunk 10–20 → lines in time order; chunk line equals `00:00–00:10 opens kettle [fill_kettle]`; agent line `00:12 AGENT: "..."`.
  - `test_json_roundtrip`.
  - `test_wake_policy_blocks_during_cooldown` — ask at 0; `can_ask(30, False)` False; `can_ask(61, False)` True; `can_ask(100, True)` False.
  - `test_nudge_only_for_none_and_respects_cooldown` — `("one")` False; `("none")` at 0 True; at 10 False; at 21 True.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. **Step 5:** Commit `feat: timeline and wake policies`.

### Task 3: Cheap signals

**Files:** Create `app/signals.py`, `tests/test_signals.py`

**Interfaces:**
- Produces:
  - `signals.visual_motion(jpegs: list[bytes]) -> float` — mean abs difference of consecutive 64×64 grayscale frames, scaled 0–1.
  - `signals.imu_energy(samples: list[dict]) -> float | None` — mean gyro magnitude (rad/s) from samples `{"t","ax","ay","az","gx","gy","gz"}`; `None` if empty.
  - `signals.imu_mismatch(visual: float, imu: float | None) -> bool` — True when `imu is not None and visual > VISUAL_MOVING (0.08) and imu < IMU_STILL (0.15)`. Constants module-level for tuning on the day.
  - `signals.HashStore(path: Path)`: `check_and_add(session_id: str, t: float, jpeg: bytes) -> list[dict]` — pHash; returns prior matches from **other** sessions with Hamming ≤ 6; persists to JSON.

- [ ] **Step 1: Failing tests** (generate images with Pillow):
  - `test_visual_motion_zero_for_identical_frames`; `test_visual_motion_high_for_alternating_black_white` (> 0.5).
  - `test_imu_energy_none_when_empty`; `test_imu_mismatch_true_when_visual_moves_and_head_still`; `test_imu_mismatch_false_without_imu`.
  - `test_hashstore_matches_other_session_not_same` — add frame for s1; same frame for s1 → `[]`; for s2 → 1 match with `session_id == "s1"`; a new `HashStore` on the same path still matches (persistence).
- [ ] **Steps 2–5:** run FAIL → implement → run PASS → commit `feat: duplicate hashing and motion/IMU signals`.

### Task 4: LLM layer

**Files:** Create `app/llm.py`, `tests/test_llm.py`, `scripts/smoke_llm.py`

**Interfaces:**
- Consumes: `Task`, `Step`, `Chunk`, `Timeline`, settings.
- Produces:
  - `llm.CaptionResult` = dict with the captioner fields; `llm.MonitorDecision` dataclass with the monitor fields (Global Constraints); `MonitorDecision.default()` → `status="ok", wake=False, divergence_type="none", fraud_risk=0.0, explained=None, question=""`.
  - `class LLM(client: AsyncOpenAI, settings)`:
    - `async caption(jpegs: list[bytes], task: Task, steps: list[Step], prev_caption: str | None) -> dict` — images as `input_image` data URLs with `detail: "low"`; strict JSON schema; on any exception/timeout returns `{"caption": "(caption unavailable)", "step": "other", "confidence": 0.0, "objects": [], "hands_visible": "both", "activity": "active", "recapture_suspected": False}`.
    - `async monitor(task: Task, references: list[str], timeline: Timeline, trigger: str) -> MonitorDecision` — `trigger` is `"chunk"` or `"worker_answer"`; text only; on failure `MonitorDecision.default()`. If `references == []`, the prompt states "No reference recordings are available; judge against the step list only."
    - `async final_report(task: Task, references: list[str], timeline: Timeline) -> dict` — `{"segments": [{"start","end","step","caption","source": "model"|"worker"}], "fraud_score": 0–100, "verdict": "accept"|"review"|"reject", "reasons": [str]}`; on failure, segments built by merging consecutive chunks with the same step, `fraud_score` null, reasons `["report model unavailable"]`.
  - All calls via `client.responses.create(model=..., input=..., text={"format": {"type": "json_schema", "name": ..., "schema": ..., "strict": True}}, timeout=settings.llm_timeout_s)`; read `response.output_text`.
  - Monitor instructions must say: interrupt rarely; prefer `watch` the first time; questions ≤12 words, friendly, about what the worker is doing; `explained` judges whether the latest WORKER line plausibly explains the latest divergence (null if no answer since the last divergence); treat `recapture_suspected`, idle, wrong objects and padding (one step far longer than in references) as divergence types.

- [ ] **Step 1: Failing tests** with a fake client object exposing `responses.create` (async, records kwargs, returns object with `output_text`):
  - `test_caption_sends_images_low_detail_and_schema` — 10 jpegs → 10 `input_image` parts, each `detail == "low"`; `text.format.strict is True`; candidate step ids appear in the prompt text.
  - `test_caption_fallback_on_exception` — fake raises → `caption == "(caption unavailable)"`.
  - `test_monitor_parses_decision` and `test_monitor_default_on_bad_json`.
  - `test_monitor_prompt_order_prefix_stable` — two calls with growing timeline: the input up to the end of the references section is byte-identical.
  - `test_monitor_mentions_no_references` when `references=[]`.
  - `test_monitor_sends_no_images`.
  - `test_final_report_fallback_merges_steps` — chunks [a,a,b] → 2 segments.
- [ ] **Step 2:** run FAIL. **Step 3:** implement. **Step 4:** run PASS.
- [ ] **Step 5:** `scripts/smoke_llm.py` — makes a real caption call on 10 synthetic frames and a monitor call on a 3-line timeline, prints results and latency. Run `uv run python scripts/smoke_llm.py` → prints valid JSON for both.
- [ ] **Step 6:** Commit `feat: captioner, monitor and report LLM calls`.

### Task 5: Voice (ElevenLabs)

**Files:** Create `app/voice.py`, `tests/test_voice.py`, `scripts/smoke_voice.py`

**Interfaces:**
- Produces `class Voice(api_key, voice_id, http: httpx.AsyncClient)`:
  - `async tts(text: str) -> bytes` — `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128`, body `{"text", "model_id": "eleven_flash_v2_5"}`; returns MP3 bytes.
  - `async stt(audio: bytes, mime: str) -> str` — `POST https://api.elevenlabs.io/v1/speech-to-text` multipart `file` + `model_id` (`scribe_v1`, or newest Scribe listed by `GET /v1/models` at setup); returns stripped `text`, `""` on error.
- [ ] **Step 1:** Look up a voice id available on the account (`GET /v1/voices`) and the Scribe model id; put defaults in `config.py` / `.env.example`.
- [ ] **Step 2: Failing tests** with `httpx.MockTransport`: `test_tts_posts_flash_model_returns_bytes`; `test_stt_returns_text`; `test_stt_returns_empty_on_500`.
- [ ] **Steps 3–4:** implement → PASS.
- [ ] **Step 5:** `scripts/smoke_voice.py` — TTS "Keep your hands in view, please." to `data/smoke.mp3`, then STT it back; prints the transcript. Run → transcript ≈ the sentence.
- [ ] **Step 6:** Commit `feat: ElevenLabs TTS and STT`.

### Task 6: Session orchestrator and report

**Files:** Create `app/session.py`, `app/report.py`, `tests/test_session.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `class Session(id: str, task: Task, library, references: list[str], llm, voice, hashes: HashStore, settings, emit: Callable[[dict], Awaitable[None]], clock: Callable[[], float] = time.monotonic)`
    - `async add_frame(t: float, jpeg: bytes)` — buffers; every `chunk_frames` frames spawns `_caption_chunk(index, frames, imu)` as a task.
    - `add_imu(samples: list[dict])`
    - `async add_answer(audio: bytes, mime: str)`
    - `async finish() -> dict` — waits for in-flight work, builds report, emits `report`.
  - Captions complete concurrently but are handed to a single monitor worker that consumes **in index order** (an `asyncio.Queue` plus a reorder buffer keyed by index).
  - Per chunk, in order: build `Chunk`; run signals (motion, IMU mismatch, hash matches on the middle frame); append chunk line (append `[recapture?]` / `[duplicate of <session> mm:ss]` / `[visual motion without head motion]` tags to the caption when signals fire so the monitor sees them); emit `chunk`; hands nudge via `NudgePolicy` (TTS cached once); call `monitor(trigger="chunk")`; emit `monitor`; if `decision.wake and wake_policy.can_ask(now, pending)` → TTS question, `timeline.add_agent`, emit `agent_say {kind: "question", text, audio_b64}`, record a `Divergence(t, type, reason, question, answer=None, explained=None)`, start answer-timeout task.
  - `add_answer`: STT → text or `"(no answer)"`; `timeline.add_worker`; emit `worker_said`; `monitor(trigger="worker_answer")` → set `explained` on the open divergence; clear pending.
  - Answer timeout (20 s) with no answer → `add_worker(t, "(no answer)")`, divergence `explained=False`, clear pending.
  - `report.signal_stats(chunks, divergences, duplicate_hits, mismatch_chunks) -> dict` — `idle_fraction, hands_out_fraction, unexplained_divergences, duplicate_hits, imu_mismatch_chunks, recapture_chunks`.
  - `report.rule_score(stats) -> int` (0–100): `40*idle_fraction + 30*hands_out_fraction + 15*unexplained + 30*min(duplicate_hits,1) + 10*imu_mismatch_chunks + 10*recapture_chunks`, capped at 100.
  - Final report = `llm.final_report(...)` plus `signals: stats`, `rule_score`, `fraud_score = round(0.5*model + 0.5*rule)` (rule only if the model score is null), `divergences`, `timeline` text.
  - Events emitted (JSON): `frame {t, jpeg_b64}` (every frame, for dashboard preview), `chunk {chunk}`, `monitor {index, decision}`, `agent_say {kind, text, audio_b64}`, `worker_said {t, text}`, `report {report}`.

- [ ] **Step 1: Failing tests** with fake LLM/voice/emit and an injectable clock:
  - `test_ten_frames_make_one_chunk_and_monitor_call`.
  - `test_chunks_monitored_in_order_when_captions_finish_out_of_order` — fake caption sleeps longer for chunk 0 than 1 → monitor sees index 0 then 1.
  - `test_wake_emits_question_with_audio_and_respects_cooldown` — monitor always wakes; 3 chunks within 30 s → exactly one `agent_say` question.
  - `test_hands_none_triggers_nudge_without_monitor_wake`.
  - `test_answer_marks_divergence_explained` — fake monitor returns `explained=True` on `worker_answer` → divergence explained, worker line in timeline.
  - `test_empty_stt_records_no_answer`.
  - `test_answer_timeout_marks_unexplained`.
  - `test_caption_failure_does_not_crash_session` — fake LLM caption returns fallback → chunk recorded, session continues.
  - `tests/test_report.py`: `test_rule_score_caps_at_100`, `test_fraud_score_uses_rule_when_model_null`.
- [ ] **Steps 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: session orchestrator with wake/answer flow and report`.

### Task 7: Server, WebSocket protocol, HTTPS

**Files:** Create `app/server.py`, `scripts/make_cert.sh`, `scripts/run.sh`, `tests/test_server.py`

**Interfaces:**
- Consumes: `Session`, `load_library`, `load_reference_timelines`, `LLM`, `Voice`, `HashStore`.
- Produces:
  - `GET /` → `static/dashboard.html`; `GET /phone` → `static/phone.html`; `/static/*`.
  - `GET /api/tasks` → `[{"id","name","steps":[...], "reference_count"}]` (reload library on each call so dropped-in tasks appear without restart).
  - `WS /ws/source` — the capture client (phone or dashboard file mode). Client → server messages: `start {task_id}`, `frame {t, jpeg_b64}`, `imu {samples}`, `answer {audio_b64, mime}`, `stop`. Server → source: `agent_say` (with audio), `report`, `error {message}`.
  - `WS /ws/watch` — dashboards; receive every event the session emits plus `session_started {session_id, task}`.
  - `start` while a session is active → finish the old one first. Unknown `task_id` → `error`.
  - App factory `create_app(settings, llm_factory=..., voice_factory=...)` so tests inject fakes.
- `scripts/make_cert.sh`: `openssl req -x509 -newkey rsa:2048 -nodes -days 7 -keyout data/key.pem -out data/cert.pem -subj "/CN=shift-agent" -addext "subjectAltName=IP:<LAN IP>,DNS:localhost"`, LAN IP from `ipconfig getifaddr en0`.
- `scripts/run.sh`: makes cert if missing, then `uv run uvicorn app.server:app --host 0.0.0.0 --port 8443 --ssl-keyfile data/key.pem --ssl-certfile data/cert.pem`, and prints `https://<LAN IP>:8443/phone`.

- [ ] **Step 1: Failing tests** (`fastapi.testclient`, fakes):
  - `test_tasks_endpoint_lists_example_task`.
  - `test_source_ws_start_frames_produce_chunk_on_watch` — watcher connects, source sends `start` + 10 frames → watcher receives `session_started`, 10 `frame`, then `chunk`.
  - `test_unknown_task_returns_error`.
  - `test_stop_emits_report_to_source_and_watch`.
- [ ] **Steps 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** `scripts/run.sh`; open `https://localhost:8443/api/tasks` in the browser pane → JSON with the example task.
- [ ] **Step 6:** Commit `feat: server, websocket protocol, HTTPS run script`.

### Task 8: Phone capture page

**Files:** Create `static/capture.js`, `static/phone.html`, `static/phone.js`, `static/style.css`

**Interfaces:**
- `capture.js` (ES module) exports:
  - `class Link(url)` — WebSocket with `send(type, payload)`, `on(type, fn)`, auto-reconnect.
  - `class FrameSampler(videoEl, {fps: 1, width: 512, quality: 0.7}, onFrame(t, jpegB64))` — canvas capture, `start()`/`stop()`; `t` = seconds since start.
  - `class ImuBuffer()` — `start()` listens to `devicemotion`, `drain() -> samples`; `requestPermission()` handles iOS `DeviceMotionEvent.requestPermission`.
  - `class VoicePlayer()` — `unlock()` (play silent audio within a gesture), `play(audioB64) -> Promise` (resolves on end).
  - `class AnswerRecorder(stream)` — `record({maxMs: 8000, silenceMs: 1500}) -> Promise<{audioB64, mime}>`, RMS silence detection via `AnalyserNode`, starts after a 300 ms grace.
- Phone UI: task picker (from `/api/tasks`), big Start/Stop button, status line, last agent line. Start tap: `voice.unlock()`, `imu.requestPermission()`, `getUserMedia({video: {facingMode: "environment"}, audio: true})`, send `start`, start sampler; every frame also sends `imu {samples: imu.drain()}`. On `agent_say`: play audio; if `kind == "question"`, then `AnswerRecorder.record()` → send `answer`.
- [ ] **Step 1:** Implement.
- [ ] **Step 2: Manual verification** with `scripts/run.sh` on the laptop and the phone on the same Wi-Fi at the printed URL (accept the certificate warning): camera preview shows; the dashboard (or the server log) shows a `chunk` after ~10 s; IMU samples arrive (log count); Start works with motion permission denied as well (Review Focus 5); a forced question (temporarily set the monitor to always wake) plays aloud on the phone and the spoken reply appears as `worker_said`.
- [ ] **Step 3:** Commit `feat: phone capture page`.

### Task 9: Dashboard and video-file mode

**Files:** Create `static/dashboard.html`, `static/dashboard.js`; modify `static/style.css`

**Interfaces:** Consumes `/ws/watch`, `/ws/source`, `/api/tasks`, `capture.js`.
- Layout (one laptop screen for judges): left = live frame (latest `frame` event) with badges for hands / idle / recapture; centre = rolling timeline (chunk captions with step chips, AGENT and WORKER lines highlighted); right = monitor status (ok/watch/diverge with reason), current step + progress bar against the task's step list, fraud-risk meter (latest `fraud_risk`), agent speech bubble.
- On `report`: switch to the report view: horizontal step segmentation bar (colour per step, divergent segments marked, worker-sourced captions marked), segment list, fraud score + verdict + reasons, signal stats.
- **File mode:** "Play a video file" button → choose MP4 + task → a hidden `<video>` plays it at 1× through `FrameSampler` over `/ws/source`; the laptop mic answers questions via `AnswerRecorder`; agent audio plays on the laptop.
- [ ] **Step 1:** Implement.
- [ ] **Step 2: Manual verification** in the browser pane at `https://localhost:8443/`: in file mode with any MP4, captions appear every ~10 s, the step bar advances, and Stop shows the report view with segments and a score.
- [ ] **Step 3:** Commit `feat: dashboard and video-file mode`.

### Task 10: Reference builder

**Files:** Create `scripts/build_refs.py`, `tests/test_build_refs.py`

**Interfaces:**
- Consumes: `LLM.caption`, `candidate_steps`, `Timeline`, `Chunk`.
- Produces: `build_refs.py <task_id> [--force]` — for each `tasks/<task_id>/refs/*.mp4` without a `.timeline.json` (or all, with `--force`), decode with PyAV at 1 FPS (frame nearest each whole second), resize to 512 px wide JPEG, group into 10-frame chunks (last partial chunk kept if ≥3 frames), caption each (previous caption passed on), write `<name>.timeline.json` = `Timeline.to_json()`. Prints a per-file summary. Exits non-zero with `"No reference videos in tasks/<id>/refs/"` when there are none.
- Helper `sample_frames(path: Path, fps: int = 1, width: int = 512) -> list[tuple[float, bytes]]` is importable for tests.
- [ ] **Step 1: Failing tests:** `test_sample_frames_one_per_second` — encode a 5 s synthetic video with PyAV in a tmp dir → 5 frames with t ≈ 0..4; `test_no_refs_exits_with_message`.
- [ ] **Steps 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Run on any short MP4 placed in `tasks/example_make_tea/refs/` → timeline file written; `/api/tasks` shows `reference_count: 1`.
- [ ] **Step 6:** Commit `feat: reference timeline builder`. Add a short `README.md` section: how to add a task, build refs, run the server, open the phone page.
