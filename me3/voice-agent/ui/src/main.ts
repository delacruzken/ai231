type Readiness = Record<string, string>;

type TranscriptRow = {
  type: string;
  text: string;
  segment_id?: string;
  role?: string;
  final?: boolean;
  partial?: boolean;
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
      out[out.length - 1] = isPartial
        ? { ...r, role }
        : { ...r, role, type: "final", partial: false, final: true };
    } else {
      out.push(
        isPartial ? { ...r, role } : { ...r, role, type: r.type || "final", final: true }
      );
    }
  }
  return out;
}

function upsertTranscriptRow(row: TranscriptRow) {
  session.transcript = coalesceTranscript([...(session.transcript || []), row]);
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

  const displayRows = coalesceTranscript(session.transcript || []);
  const lines =
    displayRows.length === 0
      ? `<div class="empty">No transcript yet. Turn ON and speak, or Send text turn.</div>`
      : displayRows
          .map((r) => {
            const role = r.role || "user";
            const kind =
              r.type === "final" || r.final === true
                ? "final"
                : r.type === "partial" || r.partial === true
                  ? "partial"
                  : "final";
            const label = role === "assistant" ? "ME3" : "You";
            const live = kind === "partial" ? `<span class="live" aria-hidden="true"></span>` : "";
            return `<div class="turn ${kind} role-${escapeAttr(role)}"><span class="who">${label}</span><span class="bubble">${escapeHtml(
              r.text || ""
            )}${live}</span></div>`;
          })
          .join("");

  const muted = !!session.mic_muted;
  const sessionOn = !!session.enabled && connected;

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

    <div class="layout">
      <aside class="col col-controls">
        <div class="toggle-wrap">
          <button type="button" class="toggle ${session.enabled ? "on" : "off"}" id="toggle" ${
            connected ? "" : "disabled"
          }>
            ${session.enabled ? "ON" : "OFF"}
          </button>
          <div class="status-line state-${escapeAttr(session.state)}">status <strong>${escapeHtml(
            session.state
          )}</strong></div>
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
            ? "Capture stays running; ASR is digitally gated. TTS and text turns still work."
            : "Listening for speech while session is ON."
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
                    return `<div class="job-card status-${escapeAttr(j.status)}">
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
    const token = getToken();
    await fetch("/api/say", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ text }),
    });
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
  session = {
    state: payload.state || "off",
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
    upsertTranscriptRow(data);
    render();
    const el = document.querySelector("#transcript");
    if (el) el.scrollTop = el.scrollHeight;
  } else if (data.type === "readiness") {
    session.readiness = data.readiness || session.readiness;
    render();
  } else if (
    data.type === "notification" ||
    data.type === "notification_delivered" ||
    data.type === "tool_result" ||
    data.type === "job_update"
  ) {
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
