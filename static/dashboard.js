// Laptop dashboard for judges: live view, end-of-session report, and video-file mode.
import {
  AgentVoiceLoop, AnswerRecorder, FrameSampler, Link, VoicePlayer, wsUrl,
} from "/static/capture.js";

const $ = (id) => document.getElementById(id);

/** Tiny element builder: h("p", {class: "x"}, "text", child). */
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") for (const [p, x] of Object.entries(v)) el.style.setProperty(p, x);
    else el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
  return el;
}

function mmss(t) {
  const s = Math.max(0, Math.floor(t));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function parseClock(s) {
  return String(s).split(":").reduce((acc, part) => acc * 60 + (Number(part) || 0), 0);
}

const pct = (x) => `${Math.round(Math.max(0, Math.min(1, Number(x) || 0)) * 100)}`;

// ------------------------------------------------------------------ state

let tasks = new Map();     // id -> {id, name, steps}
let task = null;           // task of the current session
let lastChunk = null;
let lastFrameT = 0;
let currentStep = "";
let lastAgent = null;      // the bubble's agent line, for attaching the answer

async function loadTasks() {
  try {
    const res = await fetch("/api/tasks");
    if (!res.ok) throw new Error(`Task list failed to load (HTTP ${res.status})`);
    const list = await res.json();
    tasks = new Map(list.map((t) => [t.id, t]));
  } catch (e) {
    notice(e.message);
  }
  return tasks;
}

function stepIndex(id) {
  return task ? task.steps.findIndex((s) => s.id === id) : -1;
}

function stepLabel(id) {
  const s = task && task.steps.find((x) => x.id === id);
  if (s) return s.label;
  if (!id || id === "other") return "Other activity";
  const label = id.replace(/_/g, " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

/** Step colour: one per step of the claimed task, a neutral grey for anything else. */
function stepColor(id) {
  const i = stepIndex(id);
  return i >= 0 ? `var(--step-${i % 8})` : "var(--step-foreign)";
}

function chip(stepId) {
  return h("span", { class: "chip", style: { "--c": stepColor(stepId) } },
    h("i", { "aria-hidden": "true" }), stepLabel(stepId));
}

// ------------------------------------------------------------------ notices

let noticeTimer = null;
function notice(text, { sticky = false } = {}) {
  const el = $("notice");
  el.textContent = text;
  el.hidden = !text;
  clearTimeout(noticeTimer);
  if (text && !sticky) noticeTimer = setTimeout(() => { el.hidden = true; }, 10000);
}

// ------------------------------------------------------------------ live view

function resetLive() {
  lastChunk = null;
  lastFrameT = 0;
  currentStep = "";
  lastAgent = null;
  $("timeline").replaceChildren();
  $("timeline").hidden = true;
  $("timeline-empty").hidden = false;
  $("badges").replaceChildren();
  $("frame").hidden = true;
  $("frame-empty").hidden = false;
  $("clock").textContent = "00:00";
  $("bubble").hidden = true;
  if (!file) $("frame-source").textContent = "";
  setStatus(null);
  setStep("", 0);
  setRisk(0);
  showLive();
}

function showLive() {
  $("report").hidden = true;
  $("live").hidden = false;
}

function setTask(t) {
  task = t;
  $("task-name").textContent = t ? t.name : "No session yet";
  renderStepList();
}

const STATUS = {
  ok: { word: "On track", hint: "The work matches the reference recordings." },
  watch: { word: "Watching", hint: "Something looks unusual. The agent is paying attention." },
  diverge: { word: "Off track", hint: "The work differs from the reference recordings." },
};

const DIVERGENCE = {
  off_script: "Off script", wrong_task: "Looks like a different task", out_of_order: "Steps out of order",
  padding: "Padding time", idle: "Idle", recapture: "Possible screen recapture",
};

function setStatus(decision) {
  const panel = $("status-panel");
  if (!decision) {
    panel.dataset.status = "none";
    $("status-word").textContent = "Standing by";
    $("status-reason").textContent =
      "The agent compares the work with reference recordings of this task.";
    return;
  }
  const s = STATUS[decision.status] || STATUS.ok;
  panel.dataset.status = STATUS[decision.status] ? decision.status : "ok";
  $("status-word").textContent = s.word;
  const type = DIVERGENCE[decision.divergence_type];
  const reason = (decision.reason || "").trim() || s.hint;
  $("status-reason").textContent = type && decision.status !== "ok" ? `${type}. ${reason}` : reason;
}

function setStep(stepId, progress) {
  if (stepId) currentStep = stepId;
  $("step-label").textContent = currentStep ? stepLabel(currentStep) : "None yet";
  const p = pct(progress);
  $("progress-fill").style.width = `${p}%`;
  $("progress").setAttribute("aria-valuenow", p);
  renderStepList();
}

function renderStepList() {
  const now = stepIndex(currentStep);
  const items = (task ? task.steps : []).map((s, i) => {
    const state = now < 0 ? "todo" : i < now ? "done" : i === now ? "now" : "todo";
    return h("li", { class: `step is-${state}`, style: { "--c": `var(--step-${i % 8})` } },
      h("span", { class: "step-num" }, String(i + 1)), h("span", {}, s.label));
  });
  $("steps").replaceChildren(...items);
}

function setRisk(risk) {
  const p = Number(pct(risk));
  $("risk-num").textContent = String(p);
  $("risk-fill").style.width = `${p}%`;
  $("risk-meter").setAttribute("aria-valuenow", String(p));
  $("risk-meter").dataset.level = p >= 67 ? "diverge" : p >= 34 ? "watch" : "ok";
}

/** Splits the server's appended "[...]" tags off a caption. */
function splitCaption(caption) {
  const tags = [];
  const text = String(caption || "").replace(/\s*\[([^\]]+)\]/g, (_, tag) => {
    tags.push(tag);
    return "";
  }).trim();
  return { text, tags };
}

function tagLabel(tag) {
  if (tag.startsWith("recapture")) return "Possible screen recapture";
  if (tag.startsWith("duplicate of")) return "Matches an earlier session's footage";
  if (tag.startsWith("visual motion")) return "View moves but the head is still";
  return tag;
}

function appendTimeline(row) {
  const list = $("timeline");
  const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 80;
  $("timeline-empty").hidden = true;
  list.hidden = false;
  list.append(row);
  if (atBottom) list.scrollTop = list.scrollHeight;
}

function onChunk(chunk, signals) {
  lastChunk = chunk;
  const { text, tags } = splitCaption(chunk.caption);
  appendTimeline(h("li", { class: "tl tl-chunk" },
    h("time", {}, `${mmss(chunk.t_start)}–${mmss(chunk.t_end)}`),
    h("div", { class: "tl-body" },
      h("p", { class: "tl-caption" }, text),
      h("div", { class: "tl-meta" }, chip(chunk.step),
        tags.map((t) => h("span", { class: "flag" }, tagLabel(t)))))));
  renderBadges(chunk, signals || {});
}

function renderBadges(chunk, signals) {
  const hands = {
    both: ["ok", "Both hands in view"], one: ["watch", "One hand in view"],
    none: ["diverge", "Hands out of view"],
  }[chunk.hands_visible] || ["watch", "Hands unclear"];
  const badges = [
    hands,
    chunk.activity === "idle" ? ["watch", "Idle"] : ["ok", "Working"],
  ];
  if (chunk.recapture_suspected) badges.push(["diverge", "Possible screen recapture"]);
  if (signals.duplicate) badges.push(["diverge", "Matches earlier footage"]);
  if (splitCaption(chunk.caption).tags.some((t) => t.startsWith("visual motion"))) {
    badges.push(["diverge", "View moves, head still"]);
  }
  $("badges").replaceChildren(...badges.map(([level, label]) =>
    h("li", { class: "badge", "data-level": level }, label)));
}

/** Server stamps AGENT lines at the last frame of the latest chunk. */
function agentTime() {
  return lastChunk ? lastChunk.t_end - 1 : lastFrameT;
}

function onAgentSay(ev) {
  const question = ev.kind === "question";
  const who = question ? "Agent asks" : "Agent reminds";
  appendTimeline(h("li", { class: `tl tl-agent${question ? "" : " is-nudge"}` },
    h("time", {}, mmss(agentTime())),
    h("div", { class: "tl-body" }, h("p", { class: "tl-who" }, who),
      h("p", { class: "tl-quote" }, `“${ev.text}”`))));
  $("bubble").hidden = false;
  $("bubble-who").textContent = who;
  $("bubble-text").textContent = ev.text;
  $("bubble-answer").hidden = true;
  lastAgent = question ? ev : null;
}

function onWorkerSaid(ev) {
  const none = ev.text === "(no answer)";
  const text = none ? "No answer within 20 seconds" : `“${ev.text}”`;
  appendTimeline(h("li", { class: `tl tl-worker${none ? " is-none" : ""}` },
    h("time", {}, mmss(ev.t)),
    h("div", { class: "tl-body" }, h("p", { class: "tl-who" }, "Worker answers"),
      h("p", { class: "tl-quote" }, text))));
  if (lastAgent) {
    const a = $("bubble-answer");
    a.hidden = false;
    a.textContent = none ? text : `Worker: ${text}`;
    a.classList.toggle("is-none", none);
  }
}

// ------------------------------------------------------------------ report view

function renderReport(r) {
  $("live").hidden = true;
  $("report").hidden = false;
  $("report").scrollTop = 0;
  window.scrollTo(0, 0);

  const verdict = {
    accept: ["ok", "Accept"], review: ["watch", "Needs review"], reject: ["diverge", "Reject"],
  }[r.verdict] || ["watch", "Needs review"];
  $("verdict-word").textContent = verdict[1];
  $("verdict-word").dataset.level = verdict[0];
  $("report-task").textContent = task ? task.name : "";
  const score = Math.round(Number(r.fraud_score) || 0);
  $("score-num").textContent = String(score);
  $("score-num").dataset.level = score >= 67 ? "diverge" : score >= 34 ? "watch" : "ok";
  $("score-parts").textContent = r.model_fraud_score === null || r.model_fraud_score === undefined
    ? `Signal rules ${r.rule_score ?? score}`
    : `Model ${Math.round(r.model_fraud_score)} and signal rules ${r.rule_score}, averaged`;
  $("reasons").replaceChildren(...(r.reasons || []).map((x) => h("li", {}, x)));

  const segments = (r.segments || []).map((s) => ({
    ...s, a: parseClock(s.start), b: parseClock(s.end),
  }));
  const divergences = r.divergences || [];
  const hitsSegment = (s) => divergences.some((d) => d.t >= s.a && d.t < Math.max(s.b, s.a + 1));
  renderSegbar(segments, divergences, hitsSegment);
  renderLegend(segments);

  $("seglist").replaceChildren(...segments.map((s) => {
    const { text } = splitCaption(s.caption);
    return h("li", { class: "seg-row" },
      h("time", {}, `${s.start}–${s.end}`),
      h("div", { class: "tl-body" },
        h("div", { class: "tl-meta" }, chip(s.step),
          s.source === "worker" && h("span", { class: "flag is-info" }, "From the worker's answer"),
          hitsSegment(s) && h("span", { class: "flag" }, "Agent asked here")),
        h("p", { class: "seg-caption" }, text)));
  }));
  if (!segments.length) $("seglist").append(h("li", { class: "muted" }, "No segments recorded."));

  renderSignals(r.signals || {});

  $("divlist").replaceChildren(...divergences.map((d) => {
    const status = d.explained === true ? ["ok", "Explained"] : ["diverge", "Not explained"];
    return h("li", { class: "div-row" },
      h("div", { class: "div-top" }, h("time", {}, mmss(d.t)),
        h("span", { class: "div-type" }, DIVERGENCE[d.type] || "Divergence"),
        h("span", { class: "badge", "data-level": status[0] }, status[1])),
      h("p", { class: "div-q" }, `“${d.question}”`),
      h("p", { class: "div-a" }, d.answer && d.answer !== "(no answer)"
        ? `Worker: “${d.answer}”` : "No answer"));
  }));
  if (!divergences.length) $("divlist").append(h("li", { class: "muted" }, "The agent didn't need to ask anything."));
}

function renderSegbar(segments, divergences, hitsSegment) {
  const bar = $("segbar");
  bar.replaceChildren();
  if (!segments.length) return;
  const t0 = Math.min(...segments.map((s) => s.a));
  const t1 = Math.max(...segments.map((s) => s.b));
  const span = Math.max(1, t1 - t0);
  for (const s of segments) {
    const width = t1 > t0 ? ((s.b - s.a) / span) * 100 : 100 / segments.length;
    const cls = ["seg", s.source === "worker" && "is-worker", hitsSegment(s) && "is-divergent"]
      .filter(Boolean).join(" ");
    bar.append(h("div", {
      class: cls, style: { width: `${Math.max(width, 0.6)}%`, "--c": stepColor(s.step) },
      title: `${s.start}–${s.end} ${stepLabel(s.step)}`,
    }));
  }
  for (const d of divergences) {
    const left = ((d.t - t0) / span) * 100;
    if (left < 0 || left > 100) continue;
    bar.append(h("span", { class: "seg-mark", style: { left: `${left}%` }, title: `${mmss(d.t)} ${d.question}` }));
  }
  $("axis-start").textContent = mmss(t0);
  $("axis-end").textContent = mmss(t1);
}

function renderLegend(segments) {
  const seen = new Set();
  const order = [...(task ? task.steps.map((s) => s.id) : []), ...segments.map((s) => s.step)];
  const items = [];
  for (const id of order) {
    if (seen.has(id) || !segments.some((s) => s.step === id)) continue;
    seen.add(id);
    items.push(h("li", {}, chip(id)));
  }
  if (segments.some((s) => s.source === "worker")) {
    items.push(h("li", {}, h("span", { class: "legend-key is-worker" }), "Labelled from the worker's answer"));
  }
  items.push(h("li", {}, h("span", { class: "legend-key is-divergent" }), "Agent asked a question"));
  $("legend").replaceChildren(...items);
}

const chunks = (n = 0) => `${n} ${n === 1 ? "chunk" : "chunks"}`;

function renderSignals(s) {
  const rows = [
    ["Idle time", `${pct(s.idle_fraction)}%`, s.idle_fraction > 0.2],
    ["Hands out of view", `${pct(s.hands_out_fraction)}%`, s.hands_out_fraction > 0.2],
    ["Unexplained questions", s.unexplained_divergences ?? 0, s.unexplained_divergences > 0],
    ["Matches earlier footage", s.duplicate_hits ?? 0, s.duplicate_hits > 0],
    ["View moved, head still", chunks(s.imu_mismatch_chunks), s.imu_mismatch_chunks > 0],
    ["Possible screen recapture", chunks(s.recapture_chunks), s.recapture_chunks > 0],
  ];
  $("signals").replaceChildren(...rows.map(([label, value, flagged]) =>
    h("div", { class: "signal", "data-level": flagged ? "watch" : "ok" },
      h("dt", {}, label), h("dd", {}, String(value)))));
}

// ------------------------------------------------------------------ watch link

const watch = new Link(wsUrl("/ws/watch"));

let connected = false;
let session = "none"; // none | live | ended

function renderConn() {
  const [state, text] = !connected ? ["down", "Reconnecting"]
    : session === "live" ? ["up", "Live"]
      : session === "ended" ? ["idle", "Session ended"] : ["idle", "Ready"];
  $("conn").dataset.state = state;
  $("conn").textContent = text;
}

watch.on("link:open", () => { connected = true; renderConn(); });
watch.on("link:close", () => { connected = false; renderConn(); });

watch.on("session_started", async (ev) => {
  session = "live";
  renderConn();
  if (!tasks.has(ev.task.id)) await loadTasks();
  setTask(tasks.get(ev.task.id) || { ...ev.task, steps: [] });
  resetLive();
});

watch.on("frame", (ev) => {
  lastFrameT = ev.t;
  const img = $("frame");
  img.src = `data:image/jpeg;base64,${ev.jpeg_b64}`;
  img.hidden = false;
  $("frame-empty").hidden = true;
  $("clock").textContent = mmss(ev.t);
});

watch.on("chunk", (ev) => onChunk(ev.chunk, ev.signals));
watch.on("monitor", (ev) => {
  const d = ev.decision || {};
  setStatus(d);
  setStep(d.current_step, d.progress);
  setRisk(d.fraud_risk);
});
watch.on("agent_say", onAgentSay);
watch.on("worker_said", onWorkerSaid);
watch.on("report", (ev) => {
  session = "ended";
  renderConn();
  renderReport(ev.report || {});
});

// ------------------------------------------------------------------ video-file mode

const voice = new VoicePlayer();
let source = null;   // Link to /ws/source, created on first use
let file = null;     // {video, sampler, loop, stream, url}

function fileLink() {
  if (source) return source;
  source = new Link(wsUrl("/ws/source"));
  source.on("agent_say", (ev) => { if (file && file.loop) file.loop.handle(ev); });
  source.on("report", () => endFile());
  source.on("error", (ev) => notice(ev.message || "The server reported an error."));
  source.on("link:close", () => {
    if (file) {
      endFile();
      notice("Lost the connection to the server, so the video session ended.", { sticky: true });
    }
  });
  return source;
}

function openFileDialog() {
  const sel = $("file-task");
  sel.replaceChildren(...[...tasks.values()].map((t) => new Option(t.name, t.id)));
  if (task && tasks.has(task.id)) sel.value = task.id;
  $("file-start").disabled = !tasks.size;
  if (!tasks.size) notice("No tasks found. Add a task folder under tasks/ and reload.");
  $("file-dialog").showModal();
}

async function startFile() {
  const chosen = $("file-input").files[0];
  const taskId = $("file-task").value;
  if (!chosen || !taskId) return;
  $("file-dialog").close();
  notice("");
  const unlocking = voice.unlock(); // inside the click, so agent audio can play later
  const link = fileLink();

  const video = h("video", { class: "file-video", muted: true, playsinline: true });
  video.muted = true; // the laptop mic must not hear the video's own sound
  const url = URL.createObjectURL(chosen);
  video.src = url;
  document.body.append(video);
  file = { video, url, sampler: null, loop: null, stream: null };
  setFileUi(true, chosen.name);

  try {
    await unlocking;
    let recorder = null;
    try {
      file.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      recorder = new AnswerRecorder(file.stream, voice.context);
    } catch {
      notice("Microphone blocked, so the agent's questions will go unanswered.", { sticky: true });
    }
    file.loop = new AgentVoiceLoop({
      voice,
      recorder,
      onAnswer: (a) => link.send("answer", { audio_b64: a.audioB64, mime: a.mime }),
      onState: (s) => setFileState(s),
      onError: (e) => notice(e.message || String(e)),
    });
    await link.ready();
    if (!file) return;
    link.send("start", { task_id: taskId });
    file.sampler = new FrameSampler(video, { fps: 1, width: 512, quality: 0.7 },
      (t, jpeg) => link.send("frame", { t, jpeg_b64: jpeg }));
    await video.play();
    file.sampler.start();
    video.addEventListener("ended", () => stopFile(), { once: true });
  } catch (e) {
    notice(`Couldn't play the video: ${e.message || e}`, { sticky: true });
    if (link.isOpen) link.send("stop");
    endFile();
  }
}

/** Ends the video early (or at its end) and asks the server for the report. */
function stopFile() {
  if (!file) return;
  if (source) source.send("stop");
  endFile();
  $("frame-source").textContent = "Building the report…";
}

function endFile() {
  if (!file) return;
  const f = file;
  file = null;
  if (f.sampler) f.sampler.stop();
  if (f.loop) f.loop.stop();
  if (f.stream) f.stream.getTracks().forEach((t) => t.stop());
  f.video.pause();
  f.video.remove();
  URL.revokeObjectURL(f.url);
  setFileUi(false);
}

function setFileUi(on, name = "") {
  $("file-btn").hidden = on;
  $("stop-btn").hidden = !on;
  $("frame-source").textContent = on ? `Playing ${name} from this laptop` : "";
  if (file) file.label = $("frame-source").textContent;
}

function setFileState(s) {
  if (!file) return;
  const extra = { speaking: "Agent speaking", listening: "Listening on the laptop microphone" }[s];
  $("frame-source").textContent = extra ? `${file.label}. ${extra}…` : file.label;
}

$("file-btn").addEventListener("click", async () => {
  if (!tasks.size) await loadTasks();
  openFileDialog();
});
$("file-cancel").addEventListener("click", () => $("file-dialog").close());
$("file-form").addEventListener("submit", (e) => {
  e.preventDefault();
  startFile();
});
$("stop-btn").addEventListener("click", stopFile);
$("back-btn").addEventListener("click", () => resetLive());

loadTasks();
