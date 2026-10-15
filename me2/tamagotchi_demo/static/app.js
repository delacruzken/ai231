(() => {
  const MANILA = "Asia/Manila";
  const app = document.getElementById("app");
  const enableBtn = document.getElementById("enable-sound");
  const soundPrompt = document.getElementById("sound-prompt");
  const listenPill = document.getElementById("listen-pill");
  const captionEl = document.getElementById("caption");
  const clockText = document.getElementById("clock-text");
  const dateText = document.getElementById("date-text");
  const weatherLabel = document.getElementById("weather-label");
  const weatherWidget = document.getElementById("weather-widget");
  const trackLabel = document.getElementById("track-label");
  const volFill = document.getElementById("vol-fill");
  const tempText = document.getElementById("temp-text");
  const msgBubble = document.getElementById("msg-bubble");
  const msgText = document.getElementById("msg-text");
  const reminderList = document.getElementById("reminder-list");
  const tableAlarmTime = document.getElementById("table-alarm-time");
  const tableAlarmAmPm = document.getElementById("table-alarm-ampm");
  const tableTimerTime = document.getElementById("table-timer-time");
  const hourHand = document.getElementById("hour-hand");
  const minuteHand = document.getElementById("minute-hand");

  let audioCtx = null;
  let soundReady = false;
  let timerEndAt = null;
  let timerTickId = null;
  let timerSourceSlot = "";
  let manilaWx = {
    key: "cloudy",
    label: "Partly cloudy to cloudy in Metro Manila",
    spoken: "Partly cloudy to cloudy in Metro Manila, with isolated showers possible",
  };

  function showSoundPrompt() {
    if (soundPrompt) soundPrompt.hidden = false;
    if (enableBtn) enableBtn.hidden = false;
  }

  function syncSoundReady() {
    soundReady = !!(audioCtx && audioCtx.state === "running");
    if (soundReady) dismissSoundPrompt();
    else showSoundPrompt();
    return soundReady;
  }

  function ensureAudio() {
    if (!audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AC();
    }
    if (audioCtx.state === "suspended") {
      // resume() is async; only mark ready when it actually runs.
      audioCtx.resume().then(syncSoundReady).catch(() => showSoundPrompt());
    }
    return syncSoundReady();
  }

  function tone(freq, start, dur, type, peak) {
    if (!audioCtx || audioCtx.state !== "running") return;
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, start);
    g.gain.setValueAtTime(0.0001, start);
    g.gain.exponentialRampToValueAtTime(peak, start + 0.018);
    g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
    o.connect(g);
    g.connect(audioCtx.destination);
    o.start(start);
    o.stop(start + dur + 0.02);
  }

  function postAudioGate(type, ms) {
    const body = { type, ts: Date.now() / 1000 };
    if (ms != null) body.ms = ms;
    fetch("/event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch(() => { /* ignore */ });
  }

  function estimateSpeechMs(text) {
    const words = String(text || "").trim().split(/\s+/).filter(Boolean).length;
    // Match assistant duck/mute leeway (browser TTS is often slow).
    return Math.min(14000, Math.max(2200, words * 550 + 1000));
  }

  function playBeep(kind) {
    ensureAudio();
    const busyMs = kind === "timeout" ? 480 : kind === "reject" ? 600 : 700;
    postAudioGate("audio_busy", busyMs);
    const play = () => {
      if (!audioCtx || audioCtx.state !== "running") {
        showSoundPrompt();
        setTimeout(() => postAudioGate("audio_idle"), busyMs);
        return;
      }
      const t0 = audioCtx.currentTime;
      if (kind === "reject") {
        // Low falling error (didn't understand).
        tone(392, t0, 0.16, "sine", 0.16);
        tone(247, t0 + 0.14, 0.28, "triangle", 0.18);
      } else if (kind === "timeout") {
        // Soft double blip — listen window ended, back to wake (not an error).
        tone(520, t0, 0.1, "sine", 0.11);
        tone(520, t0 + 0.16, 0.12, "sine", 0.09);
      } else {
        // Rising wake chime.
        tone(784, t0, 0.14, "triangle", 0.2);
        tone(1175, t0 + 0.13, 0.22, "triangle", 0.22);
        tone(1568, t0 + 0.26, 0.18, "sine", 0.1);
      }
      setTimeout(() => postAudioGate("audio_idle"), busyMs);
    };
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume().then(() => { syncSoundReady(); play(); }).catch(play);
    } else {
      play();
    }
  }

  function beepWake() { playBeep("wake"); }
  function beepReject() { playBeep("reject"); }
  function beepTimeout() { playBeep("timeout"); }

  function speak(text) {
    if (!text || !window.speechSynthesis) return;
    ensureAudio();
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.05;
    u.pitch = 1.0;
    const ms = estimateSpeechMs(text);
    postAudioGate("audio_busy", ms);
    u.onend = () => postAudioGate("audio_idle");
    u.onerror = () => postAudioGate("audio_idle");
    window.speechSynthesis.speak(u);
  }

  function manilaParts(now = new Date()) {
    const parts = new Intl.DateTimeFormat("en-GB", {
      timeZone: MANILA,
      hour: "numeric",
      minute: "numeric",
      hour12: false,
      weekday: "short",
      month: "numeric",
      day: "numeric",
    }).formatToParts(now);
    const get = (type) => parts.find((p) => p.type === type)?.value;
    return {
      hour: Number(get("hour")),
      minute: Number(get("minute")),
      weekday: get("weekday"),
      month: Number(get("month")),
      day: Number(get("day")),
    };
  }

  function skyFromHour(h) {
    if (h >= 5 && h < 8) return "dawn";
    if (h >= 8 && h < 17) return "day";
    if (h >= 17 && h < 20) return "dusk";
    return "night";
  }

  function weatherKey(raw) {
    const w = String(raw || "fair").toLowerCase();
    if (w.includes("thunder") || w.includes("rain") || w.includes("storm") || w.includes("shower")) return "rain";
    if (w.includes("sun") || w.includes("clear")) return "sunny";
    if (w.includes("cloud") || w.includes("overcast") || w.includes("fog")) return "cloudy";
    return "fair";
  }

  function applyManilaWeather(wx) {
    if (!wx) return;
    manilaWx = { ...manilaWx, ...wx };
    const key = weatherKey(wx.key || wx.weather || wx.label);
    manilaWx.key = key;
    weatherWidget.dataset.weather = key;
    app.dataset.weather = key;
    weatherLabel.textContent = wx.label || wx.spoken || "Metro Manila";
  }

  function manilaClockLine() {
    return new Date().toLocaleTimeString("en-PH", {
      timeZone: MANILA,
      hour: "numeric",
      minute: "2-digit",
    });
  }

  function tickClock() {
    const p = manilaParts();
    clockText.textContent = new Date().toLocaleTimeString("en-PH", {
      timeZone: MANILA,
      hour: "numeric",
      minute: "2-digit",
    });
    dateText.textContent = `${p.weekday}. ${p.month}/${p.day}`;
    app.dataset.sky = skyFromHour(p.hour);
    if (hourHand && minuteHand) {
      hourHand.setAttribute("transform", `rotate(${(p.hour % 12) * 30 + p.minute * 0.5} 386 175)`);
      minuteHand.setAttribute("transform", `rotate(${p.minute * 6} 386 175)`);
    }
  }
  tickClock();
  setInterval(tickClock, 1000);

  async function refreshManilaWeather() {
    try {
      const r = await fetch("/weather");
      if (!r.ok) throw new Error("weather http");
      applyManilaWeather(await r.json());
    } catch (_) {
      applyManilaWeather(manilaWx);
    }
  }
  refreshManilaWeather();
  setInterval(refreshManilaWeather, 10 * 60 * 1000);

  function applyState(state, opts = {}) {
    if (!state) return;
    app.classList.toggle("lights-on", !!state.lights);
    app.classList.toggle("lights-off", !state.lights);
    app.classList.remove("music-playing", "music-paused", "music-stopped");
    app.classList.add(`music-${state.music || "stopped"}`);
    app.classList.toggle("listening", !!state.listening);
    const ringing = !!opts.showPhone || (!!opts.fromHello && state.phone === "ringing");
    app.classList.toggle("phone-ringing", ringing);
    app.classList.toggle("show-time", !!state.show_time);
    app.classList.toggle("curtains-open", !!state.curtains_open);
    app.classList.toggle("show-reminders", (state.reminders || []).length > 0);
    app.classList.toggle("ac-on", !!state.ac);
    app.dataset.color = state.color || "warm";

    listenPill.hidden = !state.listening;

    let caption = state.caption || "";
    if (state.listening && (!caption || caption === "Listening…")) {
      caption = "Yes? I'm listening!";
    }
    captionEl.textContent = caption;

    if (state.live_weather || state.weather_label || state.weather_spoken) {
      applyManilaWeather({
        key: state.live_weather || manilaWx.key,
        label: state.weather_label || manilaWx.label,
        spoken: state.weather_spoken || manilaWx.spoken,
      });
    }

    trackLabel.textContent = `K.K. ${state.track || 1}`;

    // AC digital readout (always shows set temp; glows when AC is on).
    const t = Number(state.temperature);
    tempText.textContent = `${Number.isFinite(t) ? t : 24}°`;

    const vol = Math.max(0, Math.min(10, Number(state.volume) || 0));
    const h = 3 + vol * 1.4;
    volFill.setAttribute("height", String(h));
    volFill.setAttribute("y", String(20 - h));

    // Table timer: start / refresh live countdown from TIMER slot.
    const timerSlot = (state.timer || "").trim();
    if (timerSlot) {
      startTableTimer(timerSlot);
    }

    // Table alarm: always present; lights up when an alarm time is set.
    const alarmSlot = (state.alarm || "").trim();
    app.classList.toggle("alarm-set", !!alarmSlot);
    if (alarmSlot) {
      const parts = formatAlarmFace(alarmSlot);
      if (tableAlarmTime) tableAlarmTime.textContent = parts.time;
      if (tableAlarmAmPm) tableAlarmAmPm.textContent = parts.ampm;
    } else {
      if (tableAlarmTime) tableAlarmTime.textContent = "--:--";
      if (tableAlarmAmPm) tableAlarmAmPm.textContent = "";
    }

    reminderList.innerHTML = "";
    (state.reminders || []).forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      reminderList.appendChild(li);
    });

    if (opts.showMessage && state.message) {
      msgText.textContent = state.message;
      msgBubble.hidden = false;
      setTimeout(() => {
        msgBubble.hidden = true;
      }, 4000);
    } else if (!state.message) {
      msgBubble.hidden = true;
    }

    // Brightness drives the lamp only (glow / beam / room wash), not the whole den.
    const b = Math.max(0, Math.min(100, Number(state.brightness) || 60));
    app.style.setProperty("--lamp-level", String(b / 100));
  }

  /** Parse "8:00 AM" / "9:00 PM" into table-clock face parts. */
  function formatAlarmFace(slot) {
    const m = String(slot).trim().match(/(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?/i);
    if (!m) return { time: slot.slice(0, 5) || "8:00", ampm: "AM" };
    const hh = m[1];
    const mm = m[2] || "00";
    const ap = (m[3] || "AM").toUpperCase();
    return { time: `${hh}:${mm}`, ampm: ap };
  }

  /** Parse "10 seconds" / "1 minute" / "5 minutes" → total seconds. */
  function parseTimerSeconds(slot) {
    const s = String(slot || "").trim().toLowerCase();
    if (!s) return 0;
    let total = 0;
    const min = s.match(/(\d+)\s*min/);
    const sec = s.match(/(\d+)\s*sec/);
    if (min) total += parseInt(min[1], 10) * 60;
    if (sec) total += parseInt(sec[1], 10);
    if (!min && !sec) {
      const n = s.match(/(\d+)/);
      if (n) total = parseInt(n[1], 10); // bare number → seconds
    }
    return Math.max(0, total);
  }

  function formatMMSS(totalSec) {
    const s = Math.max(0, Math.floor(totalSec));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}:${String(r).padStart(2, "0")}`;
  }

  function tickTableTimer() {
    if (!timerEndAt || !tableTimerTime) return;
    const left = Math.max(0, Math.ceil((timerEndAt - Date.now()) / 1000));
    tableTimerTime.textContent = formatMMSS(left);
    if (left <= 0) {
      clearInterval(timerTickId);
      timerTickId = null;
      timerEndAt = null;
      // Brief lit "done" then dim.
      setTimeout(() => {
        app.classList.remove("timer-set");
        timerSourceSlot = "";
      }, 1800);
    }
  }

  function startTableTimer(slot) {
    const secs = parseTimerSeconds(slot);
    if (!secs) return;
    // Restart only when the spoken slot changes (or first start).
    if (slot === timerSourceSlot && timerEndAt && Date.now() < timerEndAt) return;
    timerSourceSlot = slot;
    timerEndAt = Date.now() + secs * 1000;
    app.classList.add("timer-set");
    tickTableTimer();
    clearInterval(timerTickId);
    timerTickId = setInterval(tickTableTimer, 200);
  }

  function onEvent(ev) {
    if (ev.state) {
      applyState(ev.state, {
        showMessage: ev.type === "command" && ev.intent === "MESSAGE",
        showPhone: ev.type === "command" && ev.intent === "CALL",
        fromHello: ev.type === "hello",
      });
    }

    if (ev.type === "wake") {
      msgBubble.hidden = true;
      if (window.speechSynthesis) window.speechSynthesis.cancel();
      postAudioGate("audio_idle");
      beepWake();
      return;
    }
    if (ev.type === "reset") {
      msgBubble.hidden = true;
      if (window.speechSynthesis) window.speechSynthesis.cancel();
      postAudioGate("audio_idle");
      // Full room blank — stop any live table countdown.
      clearInterval(timerTickId);
      timerTickId = null;
      timerEndAt = null;
      timerSourceSlot = "";
      app.classList.remove("timer-set", "alarm-set");
      if (tableTimerTime) tableTimerTime.textContent = "0:00";
      if (tableAlarmTime) tableAlarmTime.textContent = "--:--";
      if (tableAlarmAmPm) tableAlarmAmPm.textContent = "";
      return;
    }
    if (ev.type === "listen_end") {
      // No command before the listen window closed — soft cue, not reject.
      if (window.speechSynthesis) window.speechSynthesis.cancel();
      captionEl.textContent = "";
      beepTimeout();
      return;
    }
    if (ev.type === "reject") {
      const line = (ev.state && ev.state.caption) || "I did not understand that";
      captionEl.textContent = line;
      if (window.speechSynthesis) window.speechSynthesis.cancel();
      beepReject();
      return;
    }
    if (ev.type === "command" && ev.spoken) {
      let line = ev.spoken;
      if (ev.intent === "TIME") {
        line = `It's ${manilaClockLine()} in Manila`;
        captionEl.textContent = line;
      }
      if (ev.intent === "WEATHER") {
        line = ev.state?.weather_spoken || manilaWx.spoken;
        captionEl.textContent = line;
      }
      speak(line);
      if (ev.intent === "CALL") {
        setTimeout(() => {
          app.classList.remove("phone-ringing");
        }, 8000);
      }
    }
  }

  function dismissSoundPrompt() {
    if (enableBtn) enableBtn.hidden = true;
    if (soundPrompt) soundPrompt.hidden = true;
  }

  function unlockSound({ chime = false } = {}) {
    try {
      ensureAudio();
      if (audioCtx && audioCtx.state === "suspended") {
        audioCtx.resume().then(() => {
          syncSoundReady();
          if (chime) beepWake();
        }).catch(() => showSoundPrompt());
      } else {
        syncSoundReady();
        if (chime) beepWake();
      }
    } catch (_) {
      showSoundPrompt();
    }
  }

  // Browsers block AudioContext until a user gesture.
  function onFirstGesture() {
    unlockSound({ chime: true });
    window.removeEventListener("pointerdown", onFirstGesture, true);
    window.removeEventListener("keydown", onFirstGesture, true);
  }
  window.addEventListener("pointerdown", onFirstGesture, true);
  window.addEventListener("keydown", onFirstGesture, true);

  if (enableBtn) {
    enableBtn.addEventListener("click", () => unlockSound({ chime: true }));
  }

  const params = new URLSearchParams(location.search);
  if (params.has("nosound")) {
    dismissSoundPrompt();
  } else {
    showSoundPrompt();
  }

  const es = new EventSource("/events");
  es.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data));
    } catch (_) { /* ignore */ }
  };
  es.onerror = () => {
    // browser will retry
  };
})();
