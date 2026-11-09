type Readiness = Record<string, string>;

type TranscriptRow = {
  type: string;
  text: string;
  segment_id?: string;
  role?: string;
  final?: boolean;
  partial?: boolean;
  ts?: string;
  turn_id?: string;
};

type TypewriterState = {
  target: string;
  shownLen: number;
  final: boolean;
};

type SessionPayload = {
  state: string;
  enabled: boolean;
  readiness: Readiness;
  degraded: boolean;
  transcript: TranscriptRow[];
  last_error?: string | null;
  utterance_generation?: number;
  mic_muted?: boolean;
};

const TOKEN_KEY = "me3_ui_token";

function readStoredToken(): string {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function setToken(t: string) {
  localStorage.setItem(TOKEN_KEY, t);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

/** Prefer ?token= override; otherwise take token from same-origin bootstrap. */
async function resolveToken(): Promise<string> {
  const q = new URLSearchParams(location.search).get("token");
  if (q) {
    setToken(q);
    return q;
  }
  try {
    const res = await fetch("/api/ui-bootstrap");
    if (res.ok) {
      const data = (await res.json()) as { token?: string; require_token?: boolean };
      const boot = (data.token || "").trim();
      if (boot) {
        setToken(boot);
        return boot;
      }
      if (!data.require_token) {
        clearToken();
        return "";
      }
    }
  } catch {
    /* fall through to stored */
  }
  return readStoredToken();
}

let authToken = "";

function getToken(): string {
  return authToken || readStoredToken();
}

const app = document.querySelector("#app")!;

let session: SessionPayload = {
  state: "off",
  enabled: false,
  readiness: { audio: "degraded", vad: "degraded", asr: "degraded", tts: "degraded" },
  degraded: true,
  transcript: [],
  mic_muted: false,
};
let connected = false;
let ws: WebSocket | null = null;
let connecting = false;
let timers: { id: string; label?: string; deadline_at: string; status: string }[] = [];
let notes: { id: string; title?: string; body: string }[] = [];
type JobRow = {
  id: string;
  status: string;
  specification?: string;
  pending_user_request?: string | null;
  outcome?: string | null;
};
let jobs: JobRow[] = [];
let jobsProvider = "";
/** Optimistic: text turn submitted before session.state becomes thinking. */
let awaitingReply = false;
type Toast = { id: number; text: string; tone: "ok" | "warn" | "info"; jobId?: string };
let toasts: Toast[] = [];
let toastSeq = 0;
let highlightJobId: string | null = null;
let highlightTimer: number | null = null;
/** Character-by-character reveal for Kaki replies (keyed by turn/ts). */
const typewriters = new Map<string, TypewriterState>();
let typewriterTimer: number | null = null;
const TYPEWRITER_MS = 18;
const TYPEWRITER_CHARS = 2;

async function refreshTasks() {
  const token = getToken();
  if (!token) return;
  try {
    const [tRes, nRes, jRes] = await Promise.all([
      fetch("/api/timers", { headers: { Authorization: `Bearer ${token}` } }),
      fetch("/api/notes", { headers: { Authorization: `Bearer ${token}` } }),
      fetch("/api/jobs", { headers: { Authorization: `Bearer ${token}` } }),
    ]);
    if (tRes.ok) {
      const data = await tRes.json();
      timers = data.timers || [];
    }
    if (nRes.ok) {
      const data = await nRes.json();
      notes = data.notes || [];
    }
    if (jRes.ok) {
      const data = await jRes.json();
      jobsProvider = data.provider || "";
      jobs = data.jobs || [];
    }
    render();
  } catch {
    /* ignore */
  }
}

/** Collapse streaming partials so one utterance is one row (update in place). */
function coalesceTranscript(rows: TranscriptRow[]): TranscriptRow[] {
  const out: TranscriptRow[] = [];
  for (const r of rows) {
    const role = r.role || "user";
    const isPartial = r.type === "partial" || r.partial === true;
    const last = out[out.length - 1];
    const lastPartial =
      !!last && (last.type === "partial" || last.partial === true) && (last.role || "user") === role;
    if (lastPartial) {
      const keptTs = last.ts || r.ts;
      out[out.length - 1] = isPartial
        ? { ...r, role, ts: keptTs }
        : { ...r, role, type: "final", partial: false, final: true, ts: keptTs };
    } else if (
      last &&
      !isPartial &&
      (last.role || "user") === role &&
      !(last.type === "partial" || last.partial === true) &&
      normUtt(last.text) === normUtt(r.text)
    ) {
      // Drop duplicate finals of the same utterance (ASR + silence-commit).
      out[out.length - 1] = { ...last, ...r, role, type: "final", partial: false, final: true, ts: last.ts || r.ts };
    } else {
      out.push(
        isPartial ? { ...r, role } : { ...r, role, type: r.type || "final", final: true }
      );
    }
  }
  return out;
}

function formatClock(ts?: string): string {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
}

function normUtt(text?: string): string {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .trim();
}

function pushToast(text: string, tone: Toast["tone"] = "info", jobId?: string) {
  const id = ++toastSeq;
  toasts = [...toasts.slice(-4), { id, text, tone, jobId }];
  if (jobId) {
    highlightJobId = jobId;
    if (highlightTimer != null) window.clearTimeout(highlightTimer);
    highlightTimer = window.setTimeout(() => {
      highlightJobId = null;
      highlightTimer = null;
      render();
    }, 8000);
  }
  render();
  window.setTimeout(() => {
    toasts = toasts.filter((t) => t.id !== id);
    render();
  }, 9000);
}

function upsertTranscriptRow(row: TranscriptRow) {
  session.transcript = coalesceTranscript([...(session.transcript || []), row]);
}

/** Group chronological rows into You→Kaki exchanges (oldest → newest). */
function groupExchanges(rows: TranscriptRow[]): TranscriptRow[][] {
  const groups: TranscriptRow[][] = [];
  let cur: TranscriptRow[] = [];
  for (const r of rows) {
    const role = r.role || "user";
    if (role === "user" && cur.length) {
      groups.push(cur);
      cur = [];
    }
    cur.push(r);
  }
  if (cur.length) groups.push(cur);
  return groups;
}

/**
 * Flat display order (top → bottom):
 *   Answer3, Question3, Answer2, Question2, Answer1, Question1
 * Built by sorting — no flex `order` tricks.
 */
function flatDisplayTurns(rows: TranscriptRow[]): TranscriptRow[] {
  const groups = groupExchanges(rows);
  const scored: { row: TranscriptRow; score: number }[] = [];
  groups.forEach((group, gi) => {
    for (const row of group) {
      const isAssistant = (row.role || "user") === "assistant";
      // Newer groups get higher base; within a group assistant (answer) ranks above user.
      const score = gi * 10 + (isAssistant ? 2 : 1);
      scored.push({ row, score });
    }
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.map((s) => s.row);
}

function scrollTranscriptToLatest() {
  const el = document.querySelector("#transcript");
  if (!(el instanceof HTMLElement)) return;
  // Newest exchange is the first child — pin scroll to top after replace.
  const pin = () => {
    el.scrollTop = 0;
  };
  pin();
  requestAnimationFrame(() => {
    pin();
    requestAnimationFrame(pin);
  });
}

function assistantTwKey(r: TranscriptRow): string {
  return String(r.turn_id || r.ts || r.segment_id || r.text || "assistant").slice(0, 96);
}

function syncTypewriter(key: string, target: string, isFinal: boolean) {
  const prev = typewriters.get(key);
  if (!prev) {
    // Animate live replies; snap historical finals on reconnect/load.
    const live =
      !isFinal ||
      session.state === "speaking" ||
      session.state === "thinking" ||
      awaitingReply;
    typewriters.set(key, {
      target,
      shownLen: live ? 0 : target.length,
      final: isFinal,
    });
    if (live) ensureTypewriterTimer();
    return;
  }
  // Growing cumulative stream — keep caret; only rewind if text diverged hard.
  if (target.startsWith(prev.target.slice(0, Math.min(prev.shownLen, prev.target.length)))) {
    prev.target = target;
  } else if (!target.startsWith(prev.target) && !prev.target.startsWith(target)) {
    prev.target = target;
    prev.shownLen = 0;
  } else {
    prev.target = target;
    prev.shownLen = Math.min(prev.shownLen, target.length);
  }
  prev.final = isFinal;
  if (prev.shownLen < prev.target.length) ensureTypewriterTimer();
}

function pruneTypewriters(liveKeys: Set<string>) {
  for (const key of [...typewriters.keys()]) {
    if (!liveKeys.has(key)) typewriters.delete(key);
  }
}

function ensureTypewriterTimer() {
  if (typewriterTimer != null) return;
  typewriterTimer = window.setInterval(() => {
    let dirty = false;
    for (const tw of typewriters.values()) {
      if (tw.shownLen < tw.target.length) {
        tw.shownLen = Math.min(tw.target.length, tw.shownLen + TYPEWRITER_CHARS);
        dirty = true;
      }
    }
    const stillTyping = [...typewriters.values()].some((t) => t.shownLen < t.target.length);
    if (dirty) render();
    if (!stillTyping && typewriterTimer != null) {
      window.clearInterval(typewriterTimer);
      typewriterTimer = null;
      // One last paint so the caret drops after catching up to a final.
      if (dirty) {
        /* render() already ran */
      } else {
        render();
      }
    }
  }, TYPEWRITER_MS);
}

function renderTurn(r: TranscriptRow, displayText?: string, typing?: boolean): string {
  const role = r.role || "user";
  const kind =
    r.type === "final" || r.final === true
      ? "final"
      : r.type === "partial" || r.partial === true
        ? "partial"
        : "final";
  const label = role === "assistant" ? "Kaki" : "You";
  const text = displayText != null ? displayText : r.text || "";
  const caret = typing ? `<span class="caret" aria-hidden="true"></span>` : "";
  const live =
    !typing && kind === "partial" ? `<span class="live" aria-hidden="true"></span>` : "";
  const clock = formatClock(r.ts);
  const time = clock
    ? `<time class="ts" datetime="${escapeAttr(r.ts || "")}">${escapeHtml(clock)}</time>`
    : "";
  return `<div class="turn ${kind} role-${escapeAttr(role)}${typing ? " typing" : ""}">
    <div class="meta"><span class="who">${label}</span>${time}</div>
    <div class="bubble">${escapeHtml(text)}${caret}${live}</div>
  </div>`;
}

/** Show a pending row until Kaki starts streaming a reply. */
function shouldShowPending(): boolean {
  if (session.state !== "thinking" && !awaitingReply) return false;
  const rows = coalesceTranscript(session.transcript || []);
  let lastUserIdx = -1;
  for (let i = rows.length - 1; i >= 0; i--) {
    if ((rows[i].role || "user") === "user") {
      lastUserIdx = i;
      break;
    }
  }
  if (lastUserIdx >= 0) {
    for (let i = lastUserIdx + 1; i < rows.length; i++) {
      if ((rows[i].role || "") === "assistant" && (rows[i].text || "").trim()) {
        return false;
      }
    }
  }
  return true;
}

function pendingLabel(): string {
  if (session.state === "thinking") return "Thinking…";
  return "Working…";
}

function render() {
  const badges = Object.entries(session.readiness || {})
    .map(([k, v]) => `<span class="badge ${v}">${k}: ${v}</span>`)
    .join("");

  const timerLines =
    timers.length === 0
      ? `<div class="empty">No active timers</div>`
      : timers
          .map(
            (t) =>
              `<div class="task-row"><span class="who">timer</span> ${escapeHtml(
                t.label || t.id.slice(0, 8)
              )} · ${escapeHtml(t.deadline_at)}</div>`
          )
          .join("");
  const noteLines =
    notes.length === 0
      ? `<div class="empty">No notes yet</div>`
      : notes
          .slice(0, 8)
          .map(
            (n) =>
              `<div class="task-row"><span class="who">note</span> ${escapeHtml(
                (n.title ? n.title + ": " : "") + (n.body || "")
              )}</div>`
          )
          .join("");

  // Flat list: Answer above its Question, newest pair at top.
  const displayTurns = flatDisplayTurns(coalesceTranscript(session.transcript || []));
  const pending = shouldShowPending();
  const liveTwKeys = new Set<string>();
  const pendingTurn = pending
    ? `<div class="turn pending role-assistant" aria-live="polite" aria-busy="true">
         <div class="meta"><span class="who">Kaki</span><time class="ts">${escapeHtml(formatClock(new Date().toISOString()))}</time></div>
         <div class="bubble pending-bubble">
           <span class="spinner" aria-hidden="true"></span>
           <span class="pending-text">${escapeHtml(pendingLabel())}</span>
         </div>
       </div>`
    : "";

  const turnsHtml = displayTurns
    .map((r) => {
      if ((r.role || "user") !== "assistant") return renderTurn(r);
      const key = assistantTwKey(r);
      liveTwKeys.add(key);
      const isFinal = !(r.type === "partial" || r.partial === true);
      syncTypewriter(key, r.text || "", isFinal);
      const tw = typewriters.get(key)!;
      const typing = tw.shownLen < tw.target.length || !tw.final;
      return renderTurn(r, tw.target.slice(0, tw.shownLen), typing);
    })
    .join("");
  pruneTypewriters(liveTwKeys);
  const lines =
    displayTurns.length === 0 && !pending
      ? `<div class="empty">No transcript yet. Turn ON and speak, or Send text turn.</div>`
      : `${pendingTurn}${turnsHtml}`;

  const muted = !!session.mic_muted;
  const sessionOn = !!session.enabled && connected;

  const toastHtml =
    toasts.length === 0
      ? ""
      : `<div class="toast-stack" aria-live="polite">${toasts
          .map(
            (t) =>
              `<div class="toast tone-${escapeAttr(t.tone)}" role="status">${escapeHtml(
                t.text
              )}</div>`
          )
          .join("")}</div>`;

  app.innerHTML = `
    <header class="top">
      <div>
        <h1 class="brand">ME3</h1>
        <p class="tagline">Voice session · tools · jobs</p>
      </div>
      <p class="conn ${connected ? "" : "lost"}" id="conn">
        ${connected ? "Connected" : "Connecting…"}
      </p>
    </header>
    ${toastHtml}

    <div class="layout">
      <aside class="col col-controls">
        <div class="toggle-wrap">
          <button type="button" class="toggle ${session.enabled ? "on" : "off"}" id="toggle" ${
            connected ? "" : "disabled"
          }>
            ${session.enabled ? "ON" : "OFF"}
          </button>
          <div class="status-line state-${escapeAttr(session.state)}${
            pending ? " pending" : ""
          }">status <strong>${escapeHtml(session.state)}</strong>${
            pending ? `<span class="spinner inline" aria-hidden="true"></span>` : ""
          }</div>
        </div>

        <button
          type="button"
          class="mic-btn ${muted ? "muted" : "live"}"
          id="mic-mute"
          ${sessionOn ? "" : "disabled"}
          title="${muted ? "Unmute microphone" : "Mute microphone (session stays on)"}"
        >
          ${muted ? "Mic muted" : "Mic open"}
        </button>
        <p class="mic-hint">${
          muted
            ? "Mic muted for new questions — you can still barge in to stop Kaki mid-speech. Unmute to ask."
            : "Listening while ON. Speak over Kaki to interrupt. Headphones improve ASR + barge-in."
        }</p>

        <div class="badges">${badges}</div>

        <div class="actions">
          <button type="button" id="probe" ${sessionOn ? "" : "disabled"}>Probe TTS</button>
          <button type="button" id="say" ${sessionOn ? "" : "disabled"}>Send text turn</button>
          <button type="button" id="refresh" ${connected ? "" : "disabled"}>Refresh</button>
        </div>

        <div class="side-panel">
          <h2>Timers</h2>
          ${timerLines}
        </div>
        <div class="side-panel">
          <h2>Notes</h2>
          ${noteLines}
        </div>
        <div class="side-panel jobs-panel">
          <h2>Jobs <span class="who">${escapeHtml(jobsProvider || "—")}</span></h2>
          ${
            jobs.length === 0
              ? `<div class="empty">No jobs yet</div>`
              : jobs
                  .slice(0, 6)
                  .map((j) => {
                    const waiting = j.status === "needs_input" || j.status === "needs_approval";
                    const flash = highlightJobId === j.id ? " flash" : "";
                    return `<div class="job-card status-${escapeAttr(j.status)}${flash}">
                <div><span class="who">${escapeHtml(j.status)}</span> ${escapeHtml(
                      (j.specification || j.id).slice(0, 80)
                    )}</div>
                ${
                  j.pending_user_request
                    ? `<div class="job-ask">${escapeHtml(j.pending_user_request)}</div>`
                    : ""
                }
                ${
                  j.outcome
                    ? `<div class="empty">${escapeHtml(String(j.outcome).slice(0, 120))}</div>`
                    : ""
                }
                <div class="job-actions">
                  ${
                    waiting
                      ? `<button type="button" data-job-approve="${escapeAttr(
                          j.id
                        )}">Approve / reply</button>`
                      : ""
                  }
                  ${
                    j.status === "queued" ||
                    j.status === "running" ||
                    j.status === "needs_input" ||
                    j.status === "needs_approval"
                      ? `<button type="button" data-job-cancel="${escapeAttr(j.id)}">Cancel</button>`
                      : ""
                  }
                </div>
              </div>`;
                  })
                  .join("")
          }
        </div>
      </aside>

      <main class="col col-conversation">
        <section class="transcript-panel">
          <h2>Conversation</h2>
          <div class="transcript" id="transcript">${lines}</div>
        </section>
        <details class="diagnostics">
          <summary>Diagnostics</summary>
          <div class="token-bar">
            <input id="token" type="password" placeholder="Override UI token (optional)" value="" />
            <button type="button" id="save-token">Save override</button>
          </div>
          <pre>${escapeHtml(JSON.stringify(session, null, 2))}</pre>
        </details>
      </main>
    </div>
  `;

  document.querySelector("#save-token")?.addEventListener("click", () => {
    const v = (document.querySelector("#token") as HTMLInputElement).value.trim();
    if (v) {
      authToken = v;
      setToken(v);
    }
    void start();
  });
  document.querySelector("#toggle")?.addEventListener("click", () => {
    send({ type: "set_enabled", enabled: !session.enabled });
  });
  document.querySelector("#mic-mute")?.addEventListener("click", () => {
    send({ type: "set_mic_muted", muted: !session.mic_muted });
  });
  document.querySelector("#probe")?.addEventListener("click", () => {
    send({ type: "probe", text: "Session is on." });
  });
  document.querySelector("#say")?.addEventListener("click", async () => {
    const text = window.prompt("User text turn");
    if (!text) return;
    awaitingReply = true;
    render();
    const token = getToken();
    try {
      await fetch("/api/say", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ text }),
      });
    } catch {
      awaitingReply = false;
      render();
    }
  });
  document.querySelector("#refresh")?.addEventListener("click", () => {
    send({ type: "get_session" });
    void refreshTasks();
  });
  document.querySelectorAll("[data-job-approve]").forEach((el) => {
    el.addEventListener("click", async () => {
      const id = (el as HTMLElement).dataset.jobApprove;
      if (!id) return;
      const response = window.prompt("Reply / approval note", "approved") || "approved";
      const token = getToken();
      await fetch(`/api/jobs/${id}/respond`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ response, approve: true }),
      });
      void refreshTasks();
    });
  });
  document.querySelectorAll("[data-job-cancel]").forEach((el) => {
    el.addEventListener("click", async () => {
      const id = (el as HTMLElement).dataset.jobCancel;
      if (!id) return;
      const token = getToken();
      await fetch(`/api/jobs/${id}/cancel`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      void refreshTasks();
    });
  });

  scrollTranscriptToLatest();
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeAttr(s: string): string {
  return escapeHtml(s).replace(/'/g, "&#39;");
}

function applySession(payload: Partial<SessionPayload> & { type?: string }) {
  const nextState = payload.state || "off";
  // Drop optimistic flag once the server owns the turn (or session ends).
  // Do not clear on plain "listening" — that can race before thinking starts.
  if (nextState === "thinking" || nextState === "speaking" || nextState === "off") {
    awaitingReply = false;
  }
  session = {
    state: nextState,
    enabled: !!payload.enabled,
    readiness: payload.readiness || session.readiness,
    degraded: !!payload.degraded,
    transcript: coalesceTranscript(payload.transcript || session.transcript),
    last_error: payload.last_error,
    utterance_generation: payload.utterance_generation,
    mic_muted: !!payload.mic_muted,
  };
  render();
}

function onMessage(data: any) {
  if (data.type === "session" || data.state) {
    applySession(data);
    void refreshTasks();
  } else if (data.type === "transcript") {
    if ((data.role || "") === "assistant" && (data.text || "").trim()) {
      awaitingReply = false;
    }
    upsertTranscriptRow(data);
    render();
  } else if (data.type === "turn_done" || data.type === "turn_cancelled" || data.type === "error") {
    awaitingReply = false;
    if (data.type === "error" && data.message) {
      session.last_error = String(data.message);
    }
    render();
  } else if (data.type === "readiness") {
    session.readiness = data.readiness || session.readiness;
    render();
  } else if (data.type === "notification") {
    const msg = String(data.payload?.message || data.message || "").trim();
    if (msg) {
      const status = String(data.status || "");
      const tone: Toast["tone"] =
        status === "failed" ? "warn" : status === "completed" ? "ok" : "info";
      pushToast(msg, tone, data.job_id ? String(data.job_id) : undefined);
    }
    void refreshTasks();
  } else if (data.type === "notification_delivered" || data.type === "tool_result") {
    void refreshTasks();
  } else if (data.type === "job_update") {
    const job = data.job || {};
    const status = String(job.status || "");
    // Fallback toast if a terminal update arrives without a separate notification.
    if (
      (status === "completed" || status === "failed") &&
      !toasts.some((t) => t.jobId && job.id && t.jobId === String(job.id))
    ) {
      const label = String(job.specification || job.id || "job").split("\n")[0].slice(0, 72);
      pushToast(
        status === "completed" ? `Job finished: ${label}` : `Job failed: ${label}`,
        status === "failed" ? "warn" : "ok",
        job.id ? String(job.id) : undefined
      );
    } else if (job.id && (status === "completed" || status === "failed")) {
      highlightJobId = String(job.id);
      if (highlightTimer != null) window.clearTimeout(highlightTimer);
      highlightTimer = window.setTimeout(() => {
        highlightJobId = null;
        highlightTimer = null;
        render();
      }, 8000);
    }
    void refreshTasks();
  }
}

function send(msg: object) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

function forceOffUi() {
  connected = false;
  session = {
    ...session,
    enabled: false,
    state: "off",
  };
  render();
}

function connect() {
  if (connecting) return;
  connecting = true;
  if (ws) {
    try {
      ws.onclose = null;
      ws.close();
    } catch {
      /* ignore */
    }
    ws = null;
  }
  const token = getToken();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/session?token=${encodeURIComponent(token)}`;
  ws = new WebSocket(url);
  ws.onopen = () => {
    connecting = false;
    connected = true;
    render();
  };
  ws.onclose = () => {
    connecting = false;
    forceOffUi();
    setTimeout(() => {
      void start();
    }, 1500);
  };
  ws.onerror = () => {
    connecting = false;
    forceOffUi();
  };
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data));
    } catch {
      /* ignore */
    }
  };
}

async function start() {
  authToken = await resolveToken();
  render();
  connect();
}

render();
void start();
