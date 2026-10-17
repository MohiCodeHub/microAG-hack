// Phone capture page: camera + IMU + mic on a head strap, streaming to the laptop.
import {
  AgentVoiceLoop, AnswerRecorder, FrameSampler, ImuBuffer, Link, VoicePlayer, wsUrl,
} from "/static/capture.js";

const $ = (id) => document.getElementById(id);
const els = {
  video: $("preview"), empty: $("preview-empty"), rec: $("rec"), clock: $("clock"),
  task: $("task"), status: $("status"), error: $("error"), go: $("go"),
  agent: $("agent"), agentWho: $("agent-who"), agentText: $("agent-text"),
};

const TASK_KEY = "shift.phone.task";
const link = new Link(wsUrl("/ws/source"));
const voice = new VoicePlayer();
const imu = new ImuBuffer();

let state = "idle"; // idle | starting | recording | finishing
let stream = null;
let sampler = null;
let loop = null;
let wakeLock = null;
let motionOn = false;
let fallbackTimer = null;
let sessionId = null; // the session this phone started; events of any other are ignored

function setStatus(text) { els.status.textContent = text; }

/** Debug report to the laptop's session log (data/logs/<session>.jsonl). */
function clog(event, detail = {}) {
  try { link.send("client_log", { event, state, ...detail }); } catch { /* never break capture */ }
}

function showError(err) {
  const msg = err instanceof Error ? err.message : String(err);
  els.error.textContent = msg;
  els.error.hidden = !msg;
  if (msg) { console.error(err); clog("error_shown", { message: msg }); }
}

function setButton() {
  const labels = { idle: "Start", starting: "Starting…", recording: "Stop", finishing: "Saving…" };
  els.go.textContent = labels[state];
  els.go.classList.toggle("is-stop", state === "recording");
  els.go.disabled = state === "starting" || state === "finishing" || !els.task.value;
  els.task.disabled = state !== "idle";
}

function showAgent(who, text) {
  els.agent.hidden = false;
  els.agentWho.textContent = who;
  els.agentText.textContent = text;
}

function recordingStatus() {
  return motionOn ? "Recording. Work as normal." : "Recording without motion sensors.";
}

// ------------------------------------------------------------------ tasks

async function loadTasks() {
  try {
    const res = await fetch("/api/tasks");
    if (!res.ok) throw new Error(`Task list failed to load (HTTP ${res.status})`);
    const tasks = await res.json();
    els.task.replaceChildren(...tasks.map((t) => new Option(t.name, t.id)));
    if (!tasks.length) {
      els.task.append(new Option("No tasks on the laptop", ""));
      setStatus("Add a task folder on the laptop, then reload.");
    }
    const saved = safeGet(TASK_KEY);
    if (saved && tasks.some((t) => t.id === saved)) els.task.value = saved;
  } catch (e) {
    els.task.replaceChildren(new Option("Tasks unavailable", ""));
    showError(e);
  }
  setButton();
}

function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

// ------------------------------------------------------------------ start / stop

async function start() {
  teardown(); // never stack a second capture on a leftover one
  sessionId = null;
  state = "starting";
  setButton();
  showError("");
  els.agent.hidden = true;
  // Both must begin inside the tap for iOS: audio unlock and motion permission.
  const unlocking = voice.unlock();
  const motion = imu.requestPermission();
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access needs HTTPS. Open the https:// address the laptop printed.");
    }
    setStatus("Allow the camera and microphone…");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    els.video.srcObject = stream;
    await els.video.play().catch(() => {});
    els.empty.hidden = true;
    await unlocking;
    clog("audio_unlocked", { context: voice.context ? voice.context.state : "none" });

    motionOn = await motion;
    if (motionOn) imu.start();

    setStatus("Connecting to the laptop…");
    await link.ready();
    const taskId = els.task.value;
    safeSet(TASK_KEY, taskId);
    link.send("start", { task_id: taskId });

    loop = new AgentVoiceLoop({
      voice,
      recorder: new AnswerRecorder(stream, voice.context),
      onAnswer: (a) => {
        const sent = link.send("answer", { audio_b64: a.audioB64, mime: a.mime });
        clog("answer_recorded", { mime: a.mime, b64_chars: a.audioB64.length, sent });
      },
      onState: (s) => {
        clog("voice_state", { voice_state: s });
        if (state !== "recording") return;
        if (s === "speaking") setStatus("Agent is speaking…");
        else if (s === "listening") setStatus("Listening for your answer…");
        else setStatus(recordingStatus());
      },
      onError: showError,
    });

    sampler = new FrameSampler(els.video, { fps: 1, width: 512, quality: 0.7 }, (t, jpeg) => {
      els.clock.textContent = mmss(t);
      // IMU first, so samples land in the same chunk as the frame that follows them.
      if (motionOn) link.send("imu", { samples: imu.drain() });
      link.send("frame", { t, jpeg_b64: jpeg });
    });
    sampler.start();
    els.clock.textContent = "00:00";
    els.rec.hidden = false;
    state = "recording";
    setStatus(recordingStatus());
    keepAwake();
  } catch (e) {
    teardown();
    state = "idle";
    setStatus("Not recording.");
    showError(friendlyMediaError(e));
  }
  setButton();
}

function stop() {
  state = "finishing";
  setButton();
  link.send("stop");
  teardown();
  setStatus("Saving the session and building the report…");
  // Fall back to idle if the report never comes back.
  fallbackTimer = setTimeout(() => {
    if (state === "finishing") finished("Session sent to the laptop.");
  }, 90000);
}

function teardown() {
  if (sampler) sampler.stop();
  sampler = null;
  if (loop) loop.stop();
  loop = null;
  imu.stop();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  els.video.srcObject = null;
  els.empty.hidden = false;
  els.rec.hidden = true;
  releaseWake();
}

function finished(text) {
  clearTimeout(fallbackTimer);
  fallbackTimer = null;
  state = "idle";
  setStatus(text);
  setButton();
}

function friendlyMediaError(e) {
  const name = e && e.name;
  if (name === "NotAllowedError") {
    return "Camera or microphone was blocked. Allow both in the browser's site settings, then try again.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No usable camera or microphone was found on this device.";
  }
  return e;
}

// ------------------------------------------------------------------ screen wake lock

async function keepAwake() {
  try {
    if ("wakeLock" in navigator && state === "recording") {
      wakeLock = await navigator.wakeLock.request("screen");
    }
  } catch { /* not supported or refused; the screen may dim */ }
}

function releaseWake() {
  if (wakeLock) wakeLock.release().catch(() => {});
  wakeLock = null;
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && state === "recording") keepAwake();
});

// ------------------------------------------------------------------ server events

link.on("session_started", (ev) => {
  // Only ever a reply to our own start (also after a reconnect).
  if (state === "starting" || state === "recording") sessionId = ev.session_id;
});

link.on("agent_say", (ev) => {
  clog("agent_say_received", { kind: ev.kind, has_audio: !!ev.audio_b64,
    ev_session: ev.session_id, my_session: sessionId });
  if (state !== "recording" || ev.session_id !== sessionId) return;
  showAgent(ev.kind === "question" ? "Agent asks" : "Agent", ev.text);
  if (loop) loop.handle(ev);
});

link.on("report", (ev) => {
  if (ev.session_id !== sessionId) return; // a late report from a session we already left
  if (state === "recording") {
    // The laptop ended this session (e.g. another source started one).
    teardown();
    finished("Session ended by the laptop. Press Start to record again.");
    return;
  }
  const r = ev.report || {};
  const verdict = { accept: "accepted", review: "sent for review", reject: "rejected" }[r.verdict];
  finished(`Session saved${verdict ? `, ${verdict}` : ""}. See the laptop for the report.`);
});

link.on("error", (ev) => {
  showError(ev.message || "The laptop reported an error.");
  if (state === "finishing") finished("Not recording.");
});

link.on("link:close", () => {
  if (state === "recording") setStatus("Connection lost. Reconnecting…");
});

link.on("link:open", () => {
  // The server ends a session when its source disconnects, so start a fresh one.
  if (state === "recording") {
    link.send("start", { task_id: els.task.value });
    sampler.start();
    setStatus("Reconnected. A new session has started.");
  }
});

// ------------------------------------------------------------------ wiring

function mmss(t) {
  const s = Math.floor(t);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

els.task.addEventListener("change", setButton);
els.go.addEventListener("click", () => {
  if (state === "idle") start();
  else if (state === "recording") stop();
});

loadTasks();
