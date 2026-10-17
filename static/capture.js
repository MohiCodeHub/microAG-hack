// Shared capture classes for the phone page and the dashboard's video-file mode.
// ES module, no dependencies.

/** Absolute ws(s):// URL for a server path, matching the page's scheme. */
export function wsUrl(path) {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}${path}`;
}

/**
 * WebSocket with typed JSON messages and auto-reconnect.
 * Server messages are dispatched by their `type`; the pseudo-events
 * "link:open" and "link:close" report connection changes.
 */
export class Link {
  constructor(url) {
    this.url = url;
    this.handlers = new Map();
    this.ws = null;
    this.closed = false;
    this.retryMs = 500;
    this._connect();
  }

  get isOpen() {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  on(type, fn) {
    if (!this.handlers.has(type)) this.handlers.set(type, []);
    this.handlers.get(type).push(fn);
    return () => {
      const list = this.handlers.get(type);
      const i = list.indexOf(fn);
      if (i >= 0) list.splice(i, 1);
    };
  }

  /** Sends `{type, ...payload}`. Returns false (message dropped) when not connected. */
  send(type, payload = {}) {
    if (!this.isOpen) return false;
    this.ws.send(JSON.stringify({ type, ...payload }));
    return true;
  }

  /** Resolves once the socket is open; rejects after `timeoutMs`. */
  ready(timeoutMs = 8000) {
    if (this.isOpen) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const off = this.on("link:open", () => { clearTimeout(timer); off(); resolve(); });
      const timer = setTimeout(() => {
        off();
        reject(new Error("Can't reach the laptop server. Check both are on the same Wi-Fi."));
      }, timeoutMs);
    });
  }

  close() {
    this.closed = true;
    if (this.ws) this.ws.close();
  }

  _emit(type, data) {
    for (const fn of [...(this.handlers.get(type) || [])]) {
      try { fn(data); } catch (e) { console.error(e); }
    }
  }

  _connect() {
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => { this.retryMs = 500; this._emit("link:open"); };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg && typeof msg.type === "string") this._emit(msg.type, msg);
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this._emit("link:close");
      if (this.closed) return;
      setTimeout(() => this._connect(), this.retryMs);
      this.retryMs = Math.min(this.retryMs * 2, 5000);
    };
  }
}

/** Grabs JPEG frames from a <video> at a fixed rate; `t` = seconds since start(). */
export class FrameSampler {
  constructor(videoEl, { fps = 1, width = 512, quality = 0.7 } = {}, onFrame) {
    this.video = videoEl;
    this.fps = fps;
    this.width = width;
    this.quality = quality;
    this.onFrame = onFrame;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.timer = null;
    this.t0 = 0;
  }

  start() {
    this.stop();
    this.t0 = performance.now();
    this.timer = setInterval(() => this._grab(), 1000 / this.fps);
  }

  stop() {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
  }

  _grab() {
    const v = this.video;
    if (v.readyState < 2 || !v.videoWidth) return; // no decoded frame yet
    const w = this.width;
    const h = Math.round((v.videoHeight / v.videoWidth) * w);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.ctx.drawImage(v, 0, 0, w, h);
    const dataUrl = this.canvas.toDataURL("image/jpeg", this.quality);
    const t = Math.round(performance.now() - this.t0) / 1000;
    this.onFrame(t, dataUrl.slice(dataUrl.indexOf(",") + 1));
  }
}

const DEG = Math.PI / 180;

/**
 * Buffers devicemotion samples `{t, ax, ay, az, gx, gy, gz}` (accel m/s², gyro rad/s),
 * thinned to about 20 Hz.
 */
export class ImuBuffer {
  constructor({ hz = 20 } = {}) {
    this.minGapMs = 1000 / hz;
    this.samples = [];
    this.t0 = 0;
    this.last = -Infinity;
    this.count = 0;
    this._onMotion = (e) => this._sample(e);
  }

  /** Must run inside a user gesture on iOS. Resolves true if motion events are allowed. */
  async requestPermission() {
    if (typeof DeviceMotionEvent === "undefined") return false;
    if (typeof DeviceMotionEvent.requestPermission === "function") {
      try {
        return (await DeviceMotionEvent.requestPermission()) === "granted";
      } catch {
        return false;
      }
    }
    return true;
  }

  start() {
    this.samples = [];
    this.count = 0;
    this.t0 = performance.now();
    window.addEventListener("devicemotion", this._onMotion);
  }

  stop() {
    window.removeEventListener("devicemotion", this._onMotion);
  }

  drain() {
    const out = this.samples;
    this.samples = [];
    return out;
  }

  _sample(e) {
    const r = e.rotationRate;
    if (!r || r.alpha === null) return; // no gyroscope
    const now = performance.now();
    if (now - this.last < this.minGapMs) return;
    this.last = now;
    const a = e.acceleration && e.acceleration.x !== null ? e.acceleration
      : e.accelerationIncludingGravity || {};
    this.samples.push({
      t: Math.round(now - this.t0) / 1000,
      ax: a.x || 0, ay: a.y || 0, az: a.z || 0,
      gx: (r.beta || 0) * DEG, gy: (r.gamma || 0) * DEG, gz: (r.alpha || 0) * DEG,
    });
    this.count += 1;
  }
}

/** A short silent WAV, used to unlock audio playback inside a tap. */
function silentWavDataUrl() {
  const n = 800; // 0.1 s at 8 kHz, 8-bit mono
  const buf = new DataView(new ArrayBuffer(44 + n));
  const str = (o, s) => [...s].forEach((c, i) => buf.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); buf.setUint32(4, 36 + n, true); str(8, "WAVEfmt ");
  buf.setUint32(16, 16, true); buf.setUint16(20, 1, true); buf.setUint16(22, 1, true);
  buf.setUint32(24, 8000, true); buf.setUint32(28, 8000, true);
  buf.setUint16(32, 1, true); buf.setUint16(34, 8, true);
  str(36, "data"); buf.setUint32(40, n, true);
  for (let i = 0; i < n; i++) buf.setUint8(44 + i, 128);
  let bin = "";
  new Uint8Array(buf.buffer).forEach((b) => { bin += String.fromCharCode(b); });
  return "data:audio/wav;base64," + btoa(bin);
}

/**
 * Plays the agent's voice (MP3, base64). Call unlock() inside the Start tap so later
 * programmatic playback is allowed on iOS. unlock() also creates and resumes an
 * AudioContext (`.context`) that AnswerRecorder can share.
 */
export class VoicePlayer {
  constructor() {
    this.audio = new Audio();
    this.audio.setAttribute("playsinline", "");
    this.context = null;
    this.speaking = false;
  }

  unlock() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (Ctx && !this.context) this.context = new Ctx();
    const resumed = this.context ? this.context.resume().catch(() => {}) : Promise.resolve();
    this.audio.src = silentWavDataUrl();
    const played = this.audio.play().catch(() => {});
    return Promise.all([resumed, played]);
  }

  /** Resolves when playback ends; rejects if the browser refuses to play. */
  play(audioB64) {
    return new Promise((resolve, reject) => {
      const a = this.audio;
      const done = (err) => {
        a.onended = a.onerror = null;
        clearTimeout(guard);
        this.speaking = false;
        if (err) reject(err); else resolve();
      };
      const guard = setTimeout(() => { a.pause(); done(); }, 30000);
      this.speaking = true;
      a.onended = () => done();
      a.onerror = () => done(new Error("Couldn't play the agent's voice"));
      a.src = "data:audio/mpeg;base64," + audioB64;
      a.play().catch((e) => done(new Error(`Couldn't play the agent's voice: ${e.message}`)));
    });
  }
}

/** First MediaRecorder audio type this browser supports ("" = browser default). */
export function pickAudioMime() {
  if (typeof MediaRecorder === "undefined") return null;
  for (const m of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]) {
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

/**
 * Records one spoken answer from the stream's microphone. Stops after `silenceMs` of
 * silence following speech, after `noSpeechMs` with no speech at all, or at `maxMs`.
 * Silence detection (RMS via AnalyserNode) starts after a 300 ms grace period.
 */
export class AnswerRecorder {
  constructor(stream, audioContext = null) {
    this.stream = new MediaStream(stream.getAudioTracks());
    this.context = audioContext;
    this.recorder = null;
  }

  get recording() {
    return this.recorder !== null && this.recorder.state === "recording";
  }

  record({ maxMs = 8000, silenceMs = 1500, noSpeechMs = 5000, graceMs = 300, threshold = 0.02 } = {}) {
    const mime = pickAudioMime();
    if (mime === null) return Promise.reject(new Error("This browser can't record audio"));
    if (!this.stream.getAudioTracks().length) return Promise.reject(new Error("No microphone"));
    const rec = new MediaRecorder(this.stream, mime ? { mimeType: mime } : undefined);
    this.recorder = rec;
    const parts = [];
    rec.ondataavailable = (e) => { if (e.data && e.data.size) parts.push(e.data); };

    const level = this._levelMeter();
    const started = performance.now();
    let spoke = false;
    let lastLoud = started;
    let poll = null;
    const stop = () => { if (rec.state !== "inactive") rec.stop(); };
    const maxTimer = setTimeout(stop, maxMs);
    if (level) {
      poll = setInterval(() => {
        const now = performance.now();
        if (now - started < graceMs) return;
        if (level.rms() > threshold) { spoke = true; lastLoud = now; return; }
        if (spoke ? now - lastLoud > silenceMs : now - started > graceMs + noSpeechMs) stop();
      }, 50);
    }

    let finished = false;
    const cleanup = () => {
      if (finished) return false;
      finished = true;
      clearTimeout(maxTimer);
      if (poll) clearInterval(poll);
      if (level) level.close();
      if (this.recorder === rec) this.recorder = null;
      return true;
    };

    return new Promise((resolve, reject) => {
      rec.onstop = async () => {
        if (!cleanup()) return;
        if (rec.cancelled) { resolve(null); return; }
        const type = (rec.mimeType || mime || "audio/webm").split(";")[0];
        const blob = new Blob(parts, { type });
        try {
          resolve({ audioB64: await blobToBase64(blob), mime: type });
        } catch (e) {
          reject(e);
        }
      };
      rec.onerror = (e) => {
        if (!cleanup()) return;
        if (rec.state !== "inactive") rec.stop();
        reject(e.error || new Error("Recording failed"));
      };
      rec.start(250);
    });
  }

  /** Stops an in-progress recording; its promise resolves to null. */
  cancel() {
    if (this.recorder && this.recorder.state !== "inactive") {
      this.recorder.cancelled = true;
      this.recorder.stop();
    }
  }

  _levelMeter() {
    const ctx = this.context;
    if (!ctx || ctx.state !== "running") return null; // fall back to maxMs only
    const src = ctx.createMediaStreamSource(this.stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    src.connect(analyser);
    const data = new Float32Array(analyser.fftSize);
    return {
      rms() {
        analyser.getFloatTimeDomainData(data);
        let sum = 0;
        for (const x of data) sum += x * x;
        return Math.sqrt(sum / data.length);
      },
      close() { src.disconnect(); },
    };
  }
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).slice(String(r.result).indexOf(",") + 1));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

/**
 * Plays agent_say events one at a time and records an answer after each question.
 * `onAnswer({audioB64, mime})` is called with the recording. Nothing is recorded while the
 * agent's voice is playing, and nudges never interrupt an answer in progress.
 */
export class AgentVoiceLoop {
  constructor({ voice, recorder, onAnswer, onState = () => {}, onError = () => {} }) {
    this.voice = voice;
    this.recorder = recorder;
    this.onAnswer = onAnswer;
    this.onState = onState;
    this.onError = onError;
    this.chain = Promise.resolve();
    this.active = true;
  }

  handle(ev) {
    this.chain = this.chain.then(() => this._run(ev)).catch((e) => this.onError(e));
  }

  stop() {
    this.active = false;
    if (this.recorder) this.recorder.cancel();
  }

  async _run(ev) {
    if (!this.active) return;
    if (ev.audio_b64) {
      this.onState("speaking", ev);
      try {
        await this.voice.play(ev.audio_b64);
      } catch (e) {
        this.onError(e); // the text is still on screen; carry on
      }
    }
    if (ev.kind !== "question" || !this.active) { this.onState("idle", ev); return; }
    if (!this.recorder) {
      this.onState("idle", ev);
      throw new Error("No microphone, so the answer can't be recorded");
    }
    this.onState("listening", ev);
    const answer = await this.recorder.record();
    this.onState("idle", ev);
    if (answer && this.active) this.onAnswer(answer);
  }
}
