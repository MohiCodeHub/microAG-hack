# Live Annotation & Fraud Agent — Design

Date: 2026-10-17 · Context: one-day physical AI hackathon (London), co-hosted by MicroAGI.

## Intent

MicroAGI's Shift app pays gig workers to film egocentric task footage. Two problems they
own and haven't solved: **fraud** (padding, wrong task, recycled/synthetic footage) and
**natural-language annotation** (captions, step segmentation — currently a slow VLM pass plus
human correction). We build a voice agent that watches a recording session as it happens,
keeps a running text timeline of what the worker is doing, compares it to how that task is
normally done, and only speaks up when something diverges. The worker's spoken answer becomes
the annotation for that segment; unexplained divergences raise a fraud score.

Pitch line: *reference-conditioned, worker-verified annotation, captured while context is fresh.*

Judged on: execution, problem identification, UX, creativity/real-world impact, demo clarity.

## Goals

- Live session in a browser: camera → captions every 10 s → divergence detection → spoken
  question → spoken answer folded back into the timeline.
- End-of-session report: timestamped step segmentation with captions, fraud/quality score with
  human-readable reasons.
- A demo that reliably shows: one honest run, plus cheats (idle padding, switching task midway,
  hands out of frame, pointing the camera at a screen replaying a video).

## Non-goals

- 3D labelling (SLAM, pose, MANO/SMPL-X) — MicroAGI already does this well.
- Production scale, auth, multi-user, persistence beyond local files.
- Training any model.

## Anti-goals (failure even if it "works")

- The agent interrupts often or with long questions — it would degrade the very recordings it serves.
- The demo depends on a fragile live path with no fallback.
- API keys reach the browser or git.

## Constraints

- No hardware provided; no MicroAGI data. **Live capture is a phone on a head strap, in its browser,
  over HTTPS** (decided by the team). The laptop shows the dashboard for judges.
- **Tasks and reference videos are sourced by another teammate.** The tool must accept them as
  drop-in files (a task folder with a step list and reference MP4s); no tasks are hard-coded.
- Models (decided by the team, verified available on the account): **`gpt-5.6-luna`** for the
  captioner and monitor; **`gpt-6-astra`** available as a stronger monitor if Luna's judgement
  is weak. Both accept image input (measured ~4 s on a trivial prompt).
- **ElevenLabs** for TTS and STT.
- Cost and latency: never send a growing pile of images; images go only to the stateless captioner.

## Architecture (decided by the team in conversation)

### 1. Captioner — stateless, vision → text
- Every **10 s**, the browser sends **10 frames sampled at 1 FPS** (JPEG, downscaled) to the server.
- Server makes one **fresh** Luna call per batch (no history of images). Input: the 10 frames,
  the claimed task, the candidate step list (claimed task's steps **mixed with 1–2 distractor
  tasks' steps** so the model isn't primed to confirm), and the previous chunk's caption as one
  line of text for continuity.
- Output JSON, field order matters (free caption before step mapping):
  `{caption, step, confidence, objects[], hands_visible: both|one|none, activity: active|idle}`.

### 2. Monitor — stateful, text only
- Called once per chunk (Luna; switchable to Astra by config).
- Prompt = fixed prefix (instructions + 2–3 **reference timelines** of the same task) + the
  session timeline so far (`mm:ss–mm:ss caption [step]`, plus `mm:ss WORKER: "..."` lines for
  answers) + the newest chunk. Fixed prefix first and append-only timeline so prompt caching applies.
- Output: `{current_step, progress, status: ok|watch|diverge, divergence_type, reason, wake, question}`.
  `watch` = suspicious, give it one more chunk. `question` ≤ ~12 words.
- Worker answers are appended to the timeline, so explained variation stops re-triggering naturally.

### 3. Guardrails in code (not trusted to the model)
- Cooldown ~45 s after any question; at most one question per minute.
- **Hands nudge:** `hands_visible = none` → immediate canned TTS line ("Keep your hands in view,
  please"), no monitor wake, own cooldown.
- **Duplicate check:** perceptual hash of keyframes; matches against prior sessions' hashes flag
  recycled footage.

### 4. Voice loop
- On `wake`: TTS the question (ElevenLabs), then record the worker's mic until silence (~8 s max),
  STT (ElevenLabs), append as a WORKER line, re-run the monitor to mark the event explained/unexplained.

### 5. Session end
- One final monitor call over the full timeline → step segmentation (merged chunks with
  captions; worker answers used as captions for divergent segments) and fraud score with reasons.
  Deterministic signals (idle fraction, hands-out fraction, duplicate hits, unexplained
  divergence count) are shown alongside the model's judgement.

### 6. Reference timelines
- A script runs the **same captioner** over reference videos for each task, producing timelines in
  the same voice and format as live ones (Luna-to-Luna comparison, so wording style doesn't skew it).
- Sources: public egocentric clips and/or our own honest recordings on the day.

### Pipelining
Captioner for batch n+1 runs while the monitor handles batch n.

## My calls (builder may revisit)

- **Stack:** Python FastAPI backend (holds keys, calls OpenAI + ElevenLabs), single-page
  vanilla HTML/JS frontend served by the backend. WebSocket for frames/events. No build step.
- **Phone page** (`/phone`): pick a task, start/stop, camera + IMU + mic capture, plays the agent's
  voice (earbud). Minimal UI — the worker is busy.
- **Dashboard page** (laptop, for judges): live frame, rolling caption timeline, monitor status,
  fraud meter, agent speech bubble, then the end-of-session report (step timeline + score + reasons).
- **HTTPS for the phone:** a tunnel (cloudflared/ngrok) if available, else a self-signed cert on the
  LAN. Either way the server runs on the laptop.
- **Task file format:** `tasks/<task_id>/task.json` (name, ordered steps) + `tasks/<task_id>/refs/*.mp4`;
  a script turns the refs into reference timelines.
- **Video-file mode:** the same pipeline can ingest an uploaded MP4 instead of the webcam — used
  to build references and as the demo fallback if the live path misbehaves.
- **End-of-answer detection:** simple browser volume-based silence detection.
- **IMU/video consistency check:** the phone streams IMU, so per chunk we compare gyro/accel
  energy with frame-difference motion; a camera pointed at a replaying screen shows visual motion
  without matching head motion. Built after the core loop works.
- **Screen-replay cheat:** the captioner also returns `recapture_suspected` (screen bezel, moiré,
  glare, pixel grid visible) — the live, on-stage detector. The hash check separately catches
  re-submitting a previous session's footage.
- **Video-file mode runs in real time** with the full voice loop — the presenter answers live, so
  the fallback demos the same experience.
- **Recording and captioning continue during a question.** The question and the answer are
  timestamped lines in the timeline; there is no pause.
- ElevenLabs models: low-latency Flash TTS, Scribe STT.
- Keys live in a gitignored `.env`; never sent to the browser.

## Left to the builder

Prompt wording, exact JSON schemas, frame resolution/quality, UI styling, file layout, how the
fraud score is weighted.
