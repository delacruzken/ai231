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
  /** Idle alerts (job done / timer) — display as their own latest exchange. */
  kind?: string;
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
const ACTIVE_JOB_STATUSES = new Set([
  "queued",
  "running",
  "needs_input",
  "needs_approval",
]);

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
  instructions?: string | null;
  pending_user_request?: string | null;
  outcome?: string | null;
  updated_at?: string | null;
  updated_at_local?: string | null;
  created_at?: string | null;
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
/** Brief "Copied" label after Copy log (survives re-renders). */
let copyLogFlashUntil = 0;
let copyLogFlashTimer: number | null = null;

/** Job reply drafts survive full-DOM re-renders from WS/poll. */
let jobReplyDraft: Record<string, string> = {};
let jobBusyId: string | null = null;
let jobFormError = "";
let jobCancelConfirmId: string | null = null;
let jobPollTimer: number | null = null;
let uiClockTimer: number | null = null;

type FocusSnap = { id: string; start: number; end: number } | null;

function snapFocus(): FocusSnap {
  const el = document.activeElement;
  if (!(el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement)) return null;
  if (!el.id) return null;
  return { id: el.id, start: el.selectionStart ?? 0, end: el.selectionEnd ?? 0 };
}

function restoreFocus(snap: FocusSnap) {
  if (!snap) return;
  const el = document.getElementById(snap.id);
  if (!(el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement)) return;
  el.focus();
  try {
    el.setSelectionRange(snap.start, snap.end);
  } catch {
    /* ignore */
  }
}

function authHeaders(json = false): HeadersInit {
  const h: Record<string, string> = { Authorization: `Bearer ${getToken()}` };
  if (json) h["Content-Type"] = "application/json";
  return h;
}

async function readApiError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    if (data?.detail?.message) return String(data.detail.message);
    if (data?.error?.message) return String(data.error.message);
  } catch {
    /* ignore */
  }
  return `Request failed (${res.status})`;
}

function mergeJob(job: JobRow) {
  if (!job?.id) return;
  const idx = jobs.findIndex((j) => j.id === job.id);
  if (idx >= 0) {
    jobs = [...jobs.slice(0, idx), { ...jobs[idx], ...job }, ...jobs.slice(idx + 1)];
  } else {
    jobs = [job, ...jobs].slice(0, 20);
  }
  ensureJobPoll();
}

function hasActiveJobs(): boolean {
  return jobs.some((j) => ACTIVE_JOB_STATUSES.has(j.status));
}

function ensureJobPoll() {
  if (hasActiveJobs()) {
    if (jobPollTimer != null) return;
    jobPollTimer = window.setInterval(() => {
      void refreshTasks();
      if (!hasActiveJobs() && jobPollTimer != null) {
        window.clearInterval(jobPollTimer);
        jobPollTimer = null;
      }
    }, 4000);
  } else if (jobPollTimer != null) {
    window.clearInterval(jobPollTimer);
    jobPollTimer = null;
  }
}

function flashJob(jobId: string | undefined | null) {
  if (!jobId) return;
  highlightJobId = jobId;
  if (highlightTimer != null) window.clearTimeout(highlightTimer);
  highlightTimer = window.setTimeout(() => {
    highlightJobId = null;
    highlightTimer = null;
    render();
  }, 8000);
}

function jobTitle(j: JobRow): string {
  const line = (j.specification || j.id || "job").trim().split("\n")[0];
  if (!line) return j.id.slice(0, 8);
  if (line.length <= 160) return line;
  const cut = line.slice(0, 160);
  const sp = cut.lastIndexOf(" ");
  return `${sp > 100 ? cut.slice(0, sp) : cut}…`;
}

/**
 * Grey subtext under the job title.
 * Completed jobs: hide OpenCode chat dumps; only show a short "Build finished…" line.
 * Failed jobs: show a cleaned error (still hide UUID/`[{'type'` dumps).
 */
function jobOutcomeDisplay(outcome?: string | null, status?: string): string {
  const raw = String(outcome || "").trim();
  if (!raw) return "";
  // Raw message-part / JSON dumps from OpenCode — never show in the card.
  if (/^\s*[\[{]/.test(raw) || raw.includes("{'type'") || raw.includes('"type": "text"')) {
    return status === "completed" ? "" : "Build failed (see logs).";
  }
  const lines = raw.split(/\n/).map((l) => l.trim()).filter(Boolean);
  const uuidLine = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\/?$/i;
  const uuidHits = lines.filter((l) => uuidLine.test(l) || (/^\d+$/.test(l) && l.length <= 4)).length;
  if (uuidHits >= 2 && uuidHits >= Math.ceil(lines.length / 2)) {
    return status === "completed" ? "" : "Build failed (see logs).";
  }
  const cleaned = lines
    .filter((l) => !uuidLine.test(l) && !(/^\d+$/.test(l) && l.length <= 4))
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
  if (cleaned.length < 12) return "";
  if (status === "completed" && !/^build finished/i.test(cleaned)) {
    // Older jobs stored chat tails as outcome — don't surface them.
    return "";
  }
  return cleaned.length > 160 ? `${cleaned.slice(0, 160)}…` : cleaned;
}

function statusLabel(status: string): string {
  if (status === "completed") return "done";
  if (status === "needs_input") return "needs input";
  if (status === "needs_approval") return "needs approval";
  return status.replace(/_/g, " ");
}

/** Count-up elapsed from an ISO timestamp, e.g. "1:23" or "12s". */
function formatElapsed(iso?: string | null): string {
  const d = parseWhen(iso);
  if (!d) return "";
  const sec = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) return `${m}:${String(s).padStart(2, "0")}`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function formatDurationBetween(startIso?: string | null, endIso?: string | null): string {
  const a = parseWhen(startIso);
  const b = parseWhen(endIso);
  if (!a || !b) return "";
  const sec = Math.max(0, Math.floor((b.getTime() - a.getTime()) / 1000));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function jobTimeLabel(j: JobRow): string {
  const start = j.created_at || j.updated_at || j.updated_at_local;
  if (j.status === "queued" || j.status === "running") {
    const el = formatElapsed(start);
    return el ? `↑ ${el}` : "live";
  }
  if (j.status === "needs_input" || j.status === "needs_approval") {
    return "waiting on you";
  }
  if (j.status === "completed") {
    const took = formatDurationBetween(j.created_at, j.updated_at || j.updated_at_local);
    return took ? `took ${took}` : "done";
  }
  if (j.status === "failed") {
    return "failed";
  }
  return formatRelative(j.updated_at_local || j.updated_at || j.created_at);
}

function parseWhen(iso?: string | null): Date | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** Relative age, e.g. "2m ago". */
function formatRelative(iso?: string | null): string {
  const d = parseWhen(iso);
  if (!d) return "";
  const sec = Math.round((Date.now() - d.getTime()) / 1000);
  if (sec < 10) return "just now";
  if (sec < 60) return `${sec}s ago`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 48) return `${hr}h ago`;
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

/** Countdown or overdue for timer deadlines. */
function formatCountdown(deadlineAt: string): string {
  const d = parseWhen(deadlineAt);
  if (!d) return deadlineAt;
  const sec = Math.round((d.getTime() - Date.now()) / 1000);
  if (sec <= 0) {
    const late = Math.abs(sec);
    if (late < 60) return "due now";
    if (late < 3600) return `${Math.round(late / 60)}m overdue`;
    return `${Math.round(late / 3600)}h overdue`;
  }
  if (sec < 60) return `in ${sec}s`;
  if (sec < 3600) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return s ? `in ${m}m ${s}s` : `in ${m}m`;
  }
  if (sec < 86400) {
    const h = Math.floor(sec / 3600);
    const m = Math.round((sec % 3600) / 60);
    return m ? `in ${h}h ${m}m` : `in ${h}h`;
  }
  const days = Math.round(sec / 86400);
  return `in ${days}d`;
}

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
    ensureJobPoll();
    if (timers.length || hasActiveJobs()) ensureUiClock();
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

/** Chronological (oldest → newest) plain-text log for pasting into chat / validation. */
function formatConversationLog(rows: TranscriptRow[]): string {
  const lines: string[] = ["Conversation"];
  for (const r of coalesceTranscript(rows)) {
    const text = (r.text || "").trim();
    if (!text) continue;
    const who = (r.role || "user") === "assistant" ? "Kaki" : "You";
    const clock = formatClock(r.ts);
    lines.push(who);
    if (clock) lines.push(clock);
    lines.push(text);
  }
  return lines.join("\n");
}

async function copyConversationLog(): Promise<boolean> {
  const text = formatConversationLog(session.transcript || []);
  if (text === "Conversation") return false;
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback when clipboard permission is denied / insecure context.
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } finally {
      document.body.removeChild(ta);
    }
    return ok;
  }
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
  if (jobId) flashJob(jobId);
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
    const isNotice = (r.kind || "") === "notification" || (r.kind || "") === "clarify";
    // Notifications are standalone latest items — do not tuck them into the prior Q&A.
    if (((role === "user" && !isNotice) || isNotice) && cur.length) {
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

function renderJobsPanel(): string {
  const cards =
    jobs.length === 0
      ? `<div class="empty">No jobs yet — ask Kaki to start one.</div>`
      : jobs
          .slice(0, 8)
          .map((j) => {
            const waiting = j.status === "needs_input" || j.status === "needs_approval";
            const cancellable = ACTIVE_JOB_STATUSES.has(j.status);
            const flash = highlightJobId === j.id ? " flash" : "";
            const cardBusy = jobBusyId === j.id;
            const confirming = jobCancelConfirmId === j.id;
            const when = jobTimeLabel(j);
            const live =
              j.status === "queued" || j.status === "running"
                ? `<span class="job-live" aria-hidden="true"></span>`
                : "";
            const replyVal = jobReplyDraft[j.id] ?? "";
            return `<div class="job-card status-${escapeAttr(j.status)}${flash}" data-job-id="${escapeAttr(
              j.id
            )}">
              <div class="job-card-top">
                <span class="status-pill status-${escapeAttr(j.status)}">${live}${escapeHtml(
                  statusLabel(j.status)
                )}</span>
                ${when ? `<span class="job-when">${escapeHtml(when)}</span>` : ""}
              </div>
              <div class="job-title">${escapeHtml(jobTitle(j))}</div>
              ${
                j.pending_user_request
                  ? `<div class="job-ask">${escapeHtml(j.pending_user_request)}</div>`
                  : ""
              }
              ${
                (() => {
                  if (!(j.status === "completed" || j.status === "failed")) return "";
                  const shown = jobOutcomeDisplay(j.outcome, j.status);
                  return shown ? `<div class="job-outcome">${escapeHtml(shown)}</div>` : "";
                })()
              }
              ${
                waiting
                  ? `<div class="job-reply">
                      <textarea
                        id="job-reply-${escapeAttr(j.id)}"
                        data-job-reply="${escapeAttr(j.id)}"
                        rows="2"
                        placeholder="${
                          j.status === "needs_approval"
                            ? "Optional note, or leave blank to approve"
                            : "Your reply…"
                        }"
                        ${cardBusy ? "disabled" : ""}
                      >${escapeHtml(replyVal)}</textarea>
                      <div class="job-actions">
                        <button type="button" class="btn-primary" data-job-reply-send="${escapeAttr(
                          j.id
                        )}" ${cardBusy ? "disabled" : ""}>${
                          cardBusy ? "Sending…" : "Reply"
                        }</button>
                        <button type="button" data-job-approve="${escapeAttr(j.id)}" ${
                          cardBusy ? "disabled" : ""
                        }>Approve</button>
                      </div>
                    </div>`
                  : ""
              }
              ${
                cancellable
                  ? `<div class="job-actions job-actions-secondary">
                      ${
                        confirming
                          ? `<span class="job-confirm-label">Cancel this job?</span>
                             <button type="button" class="btn-danger" data-job-cancel-yes="${escapeAttr(
                               j.id
                             )}" ${cardBusy ? "disabled" : ""}>${
                               cardBusy ? "Cancelling…" : "Yes, cancel"
                             }</button>
                             <button type="button" data-job-cancel-no="${escapeAttr(
                               j.id
                             )}" ${cardBusy ? "disabled" : ""}>Keep</button>`
                          : `<button type="button" data-job-cancel="${escapeAttr(j.id)}" ${
                              cardBusy ? "disabled" : ""
                            }>Cancel</button>`
                      }
                    </div>`
                  : ""
              }
            </div>`;
          })
          .join("");

  const active = jobs.find((j) => ACTIVE_JOB_STATUSES.has(j.status));
  const banner = active
    ? `<div class="jobs-live-banner status-${escapeAttr(active.status)}" aria-live="polite">
         <span class="job-live" aria-hidden="true"></span>
         <strong>${escapeHtml(statusLabel(active.status))}</strong>
         <span>${escapeHtml(jobTimeLabel(active))}</span>
         <span class="jobs-live-title">${escapeHtml(jobTitle(active))}</span>
       </div>`
    : jobs.some((j) => j.status === "completed")
      ? `<div class="jobs-live-banner status-completed" aria-live="polite">
           <strong>done</strong>
           <span>latest build finished</span>
         </div>`
      : "";

  return `<div class="side-panel jobs-panel">
    <h2>Jobs <span class="who">${escapeHtml(jobsProvider || "—")}</span></h2>
    ${banner}
    ${
      jobFormError
        ? `<div class="form-error" role="alert">${escapeHtml(jobFormError)}</div>`
        : ""
    }
    <div class="job-list">${cards}</div>
  </div>`;
}

function render() {
  const focus = snapFocus();
  const badges = Object.entries(session.readiness || {})
    .map(([k, v]) => `<span class="badge ${v}">${k}: ${v}</span>`)
    .join("");

  const timerLines =
    timers.length === 0
      ? `<div class="empty">No active timers</div>`
      : timers
          .map((t) => {
            const label = t.label || t.id.slice(0, 8);
            const cd = formatCountdown(t.deadline_at);
            return `<div class="task-row"><span class="who">timer</span> ${escapeHtml(
              label
            )} · <span class="timer-countdown">${escapeHtml(cd)}</span></div>`;
          })
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
        <h1 class="brand">Kaki</h1>
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
        ${renderJobsPanel()}
      </aside>

      <main class="col col-conversation">
        <section class="transcript-panel">
          <div class="panel-heading">
            <h2>Conversation</h2>
            <button
              type="button"
              class="copy-log ${Date.now() < copyLogFlashUntil ? "copied" : ""}"
              id="copy-log"
              ${displayTurns.length === 0 ? "disabled" : ""}
              title="Copy conversation log (oldest → newest)"
            >${Date.now() < copyLogFlashUntil ? "Copied" : "Copy log"}</button>
          </div>
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

  bindUiHandlers();
  restoreFocus(focus);
  scrollTranscriptToLatest();
}

function bindUiHandlers() {
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
    try {
      await fetch("/api/say", {
        method: "POST",
        headers: authHeaders(true),
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
  document.querySelector("#copy-log")?.addEventListener("click", () => {
    void copyConversationLog().then((ok) => {
      if (!ok) {
        pushToast("Could not copy conversation log", "warn");
        return;
      }
      copyLogFlashUntil = Date.now() + 1600;
      if (copyLogFlashTimer != null) window.clearTimeout(copyLogFlashTimer);
      copyLogFlashTimer = window.setTimeout(() => {
        copyLogFlashUntil = 0;
        copyLogFlashTimer = null;
        render();
      }, 1600);
      pushToast("Conversation log copied", "ok");
    });
  });

  document.querySelectorAll("[data-job-reply]").forEach((el) => {
    el.addEventListener("input", () => {
      const id = (el as HTMLElement).dataset.jobReply;
      if (!id) return;
      jobReplyDraft[id] = (el as HTMLTextAreaElement).value;
    });
  });

  document.querySelectorAll("[data-job-reply-send]").forEach((el) => {
    el.addEventListener("click", () => {
      const id = (el as HTMLElement).dataset.jobReplySend;
      if (!id) return;
      void respondJob(id, jobReplyDraft[id] || "", false);
    });
  });
  document.querySelectorAll("[data-job-approve]").forEach((el) => {
    el.addEventListener("click", () => {
      const id = (el as HTMLElement).dataset.jobApprove;
      if (!id) return;
      const note = (jobReplyDraft[id] || "").trim() || "approved";
      void respondJob(id, note, true);
    });
  });
  document.querySelectorAll("[data-job-cancel]").forEach((el) => {
    el.addEventListener("click", () => {
      const id = (el as HTMLElement).dataset.jobCancel;
      if (!id) return;
      jobCancelConfirmId = id;
      render();
    });
  });
  document.querySelectorAll("[data-job-cancel-no]").forEach((el) => {
    el.addEventListener("click", () => {
      jobCancelConfirmId = null;
      render();
    });
  });
  document.querySelectorAll("[data-job-cancel-yes]").forEach((el) => {
    el.addEventListener("click", () => {
      const id = (el as HTMLElement).dataset.jobCancelYes;
      if (!id) return;
      void cancelJob(id);
    });
  });
}

async function respondJob(jobId: string, response: string, approve: boolean) {
  if (jobBusyId != null) return;
  jobBusyId = jobId;
  jobFormError = "";
  render();
  try {
    const res = await fetch(`/api/jobs/${jobId}/respond`, {
      method: "POST",
      headers: authHeaders(true),
      body: JSON.stringify({ response, approve }),
    });
    if (!res.ok) {
      const err = await readApiError(res);
      jobFormError = err;
      pushToast(err, "warn", jobId);
      return;
    }
    const data = await res.json();
    const job = (data.job || data) as JobRow;
    if (job?.id) mergeJob(job);
    delete jobReplyDraft[jobId];
    pushToast(approve ? "Job approved" : "Reply sent", "ok", jobId);
    void refreshTasks();
  } catch {
    jobFormError = "Could not reach the server";
    pushToast(jobFormError, "warn", jobId);
  } finally {
    jobBusyId = null;
    render();
  }
}

async function cancelJob(jobId: string) {
  if (jobBusyId != null) return;
  jobBusyId = jobId;
  jobFormError = "";
  render();
  try {
    const res = await fetch(`/api/jobs/${jobId}/cancel`, {
      method: "POST",
      headers: authHeaders(),
    });
    if (!res.ok) {
      const err = await readApiError(res);
      jobFormError = err;
      pushToast(err, "warn", jobId);
      return;
    }
    const data = await res.json();
    const job = (data.job || data) as JobRow;
    if (job?.id) mergeJob(job);
    jobCancelConfirmId = null;
    pushToast("Job cancelled", "info", jobId);
    void refreshTasks();
  } catch {
    jobFormError = "Could not reach the server";
    pushToast(jobFormError, "warn", jobId);
  } finally {
    jobBusyId = null;
    jobCancelConfirmId = null;
    render();
  }
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
    if (job.id) {
      mergeJob(job as JobRow);
      render();
    }
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
      flashJob(String(job.id));
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
    void refreshTasks();
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

function ensureUiClock() {
  if (uiClockTimer != null) return;
  // 1s tick while timers (or active jobs) need live relative labels.
  uiClockTimer = window.setInterval(() => {
    if (timers.length || hasActiveJobs()) {
      render();
    } else if (uiClockTimer != null) {
      window.clearInterval(uiClockTimer);
      uiClockTimer = null;
    }
  }, 1000);
}

render();
void start();
