(function(){const n=document.createElement("link").relList;if(n&&n.supports&&n.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))a(s);new MutationObserver(s=>{for(const o of s)if(o.type==="childList")for(const u of o.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&a(u)}).observe(document,{childList:!0,subtree:!0});function i(s){const o={};return s.integrity&&(o.integrity=s.integrity),s.referrerPolicy&&(o.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?o.credentials="include":s.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function a(s){if(s.ep)return;s.ep=!0;const o=i(s);fetch(s.href,o)}})();const O="me3_ui_token";function M(){return localStorage.getItem(O)||""}function k(e){localStorage.setItem(O,e)}function B(){localStorage.removeItem(O)}async function I(){const e=new URLSearchParams(location.search).get("token");if(e)return k(e),e;try{const n=await fetch("/api/ui-bootstrap");if(n.ok){const i=await n.json(),a=(i.token||"").trim();if(a)return k(a),a;if(!i.require_token)return B(),""}}catch{}return M()}let T="";function y(){return T||M()}const J=document.querySelector("#app");let r={state:"off",enabled:!1,readiness:{audio:"degraded",vad:"degraded",asr:"degraded",tts:"degraded"},degraded:!0,transcript:[],mic_muted:!1},f=!1,d=null,m=!1,S=[],w=[],_=[],x="";async function h(){const e=y();if(e)try{const[n,i,a]=await Promise.all([fetch("/api/timers",{headers:{Authorization:`Bearer ${e}`}}),fetch("/api/notes",{headers:{Authorization:`Bearer ${e}`}}),fetch("/api/jobs",{headers:{Authorization:`Bearer ${e}`}})]);if(n.ok&&(S=(await n.json()).timers||[]),i.ok&&(w=(await i.json()).notes||[]),a.ok){const s=await a.json();x=s.provider||"",_=s.jobs||[]}p()}catch{}}function q(e){const n=[];for(const i of e){const a=i.role||"user",s=i.type==="partial"||i.partial===!0,o=n[n.length-1];!!o&&(o.type==="partial"||o.partial===!0)&&(o.role||"user")===a?n[n.length-1]=s?{...i,role:a}:{...i,role:a,type:"final",partial:!1,final:!0}:n.push(s?{...i,role:a}:{...i,role:a,type:i.type||"final",final:!0})}return n}function U(e){r.transcript=q([...r.transcript||[],e])}function p(){var N,j,E,P,A,R;const e=Object.entries(r.readiness||{}).map(([t,c])=>`<span class="badge ${c}">${t}: ${c}</span>`).join(""),n=S.length===0?'<div class="empty">No active timers</div>':S.map(t=>`<div class="task-row"><span class="who">timer</span> ${l(t.label||t.id.slice(0,8))} · ${l(t.deadline_at)}</div>`).join(""),i=w.length===0?'<div class="empty">No notes yet</div>':w.slice(0,8).map(t=>`<div class="task-row"><span class="who">note</span> ${l((t.title?t.title+": ":"")+(t.body||""))}</div>`).join(""),a=q(r.transcript||[]),s=a.length===0?'<div class="empty">No transcript yet. Turn ON and speak, or Send text turn.</div>':a.map(t=>{const c=t.role||"user",v=t.type==="final"||t.final===!0?"final":t.type==="partial"||t.partial===!0?"partial":"final",$=c==="assistant"?"ME3":"You",z=v==="partial"?'<span class="live" aria-hidden="true"></span>':"";return`<div class="turn ${v} role-${b(c)}"><span class="who">${$}</span><span class="bubble">${l(t.text||"")}${z}</span></div>`}).join(""),o=!!r.mic_muted,u=!!r.enabled&&f;J.innerHTML=`
    <header class="top">
      <div>
        <h1 class="brand">ME3</h1>
        <p class="tagline">Voice session · tools · jobs</p>
      </div>
      <p class="conn ${f?"":"lost"}" id="conn">
        ${f?"Connected":"Connecting…"}
      </p>
    </header>

    <div class="layout">
      <aside class="col col-controls">
        <div class="toggle-wrap">
          <button type="button" class="toggle ${r.enabled?"on":"off"}" id="toggle" ${f?"":"disabled"}>
            ${r.enabled?"ON":"OFF"}
          </button>
          <div class="status-line state-${b(r.state)}">status <strong>${l(r.state)}</strong></div>
        </div>

        <button
          type="button"
          class="mic-btn ${o?"muted":"live"}"
          id="mic-mute"
          ${u?"":"disabled"}
          title="${o?"Unmute microphone":"Mute microphone (session stays on)"}"
        >
          ${o?"Mic muted":"Mic open"}
        </button>
        <p class="mic-hint">${o?"Capture stays running; ASR is digitally gated. TTS and text turns still work.":"Listening for speech while session is ON."}</p>

        <div class="badges">${e}</div>

        <div class="actions">
          <button type="button" id="probe" ${u?"":"disabled"}>Probe TTS</button>
          <button type="button" id="say" ${u?"":"disabled"}>Send text turn</button>
          <button type="button" id="refresh" ${f?"":"disabled"}>Refresh</button>
        </div>

        <div class="side-panel">
          <h2>Timers</h2>
          ${n}
        </div>
        <div class="side-panel">
          <h2>Notes</h2>
          ${i}
        </div>
        <div class="side-panel jobs-panel">
          <h2>Jobs <span class="who">${l(x||"—")}</span></h2>
          ${_.length===0?'<div class="empty">No jobs yet</div>':_.slice(0,6).map(t=>{const c=t.status==="needs_input"||t.status==="needs_approval";return`<div class="job-card status-${b(t.status)}">
                <div><span class="who">${l(t.status)}</span> ${l((t.specification||t.id).slice(0,80))}</div>
                ${t.pending_user_request?`<div class="job-ask">${l(t.pending_user_request)}</div>`:""}
                ${t.outcome?`<div class="empty">${l(String(t.outcome).slice(0,120))}</div>`:""}
                <div class="job-actions">
                  ${c?`<button type="button" data-job-approve="${b(t.id)}">Approve / reply</button>`:""}
                  ${t.status==="queued"||t.status==="running"||t.status==="needs_input"||t.status==="needs_approval"?`<button type="button" data-job-cancel="${b(t.id)}">Cancel</button>`:""}
                </div>
              </div>`}).join("")}
        </div>
      </aside>

      <main class="col col-conversation">
        <section class="transcript-panel">
          <h2>Conversation</h2>
          <div class="transcript" id="transcript">${s}</div>
        </section>
        <details class="diagnostics">
          <summary>Diagnostics</summary>
          <div class="token-bar">
            <input id="token" type="password" placeholder="Override UI token (optional)" value="" />
            <button type="button" id="save-token">Save override</button>
          </div>
          <pre>${l(JSON.stringify(r,null,2))}</pre>
        </details>
      </main>
    </div>
  `,(N=document.querySelector("#save-token"))==null||N.addEventListener("click",()=>{const t=document.querySelector("#token").value.trim();t&&(T=t,k(t)),L()}),(j=document.querySelector("#toggle"))==null||j.addEventListener("click",()=>{g({type:"set_enabled",enabled:!r.enabled})}),(E=document.querySelector("#mic-mute"))==null||E.addEventListener("click",()=>{g({type:"set_mic_muted",muted:!r.mic_muted})}),(P=document.querySelector("#probe"))==null||P.addEventListener("click",()=>{g({type:"probe",text:"Session is on."})}),(A=document.querySelector("#say"))==null||A.addEventListener("click",async()=>{const t=window.prompt("User text turn");if(!t)return;const c=y();await fetch("/api/say",{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${c}`},body:JSON.stringify({text:t})})}),(R=document.querySelector("#refresh"))==null||R.addEventListener("click",()=>{g({type:"get_session"}),h()}),document.querySelectorAll("[data-job-approve]").forEach(t=>{t.addEventListener("click",async()=>{const c=t.dataset.jobApprove;if(!c)return;const v=window.prompt("Reply / approval note","approved")||"approved",$=y();await fetch(`/api/jobs/${c}/respond`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${$}`},body:JSON.stringify({response:v,approve:!0})}),h()})}),document.querySelectorAll("[data-job-cancel]").forEach(t=>{t.addEventListener("click",async()=>{const c=t.dataset.jobCancel;if(!c)return;const v=y();await fetch(`/api/jobs/${c}/cancel`,{method:"POST",headers:{Authorization:`Bearer ${v}`}}),h()})})}function l(e){return e.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}function b(e){return l(e).replace(/'/g,"&#39;")}function F(e){r={state:e.state||"off",enabled:!!e.enabled,readiness:e.readiness||r.readiness,degraded:!!e.degraded,transcript:q(e.transcript||r.transcript),last_error:e.last_error,utterance_generation:e.utterance_generation,mic_muted:!!e.mic_muted},p()}function H(e){if(e.type==="session"||e.state)F(e),h();else if(e.type==="transcript"){U(e),p();const n=document.querySelector("#transcript");n&&(n.scrollTop=n.scrollHeight)}else e.type==="readiness"?(r.readiness=e.readiness||r.readiness,p()):(e.type==="notification"||e.type==="notification_delivered"||e.type==="tool_result"||e.type==="job_update")&&h()}function g(e){d&&d.readyState===WebSocket.OPEN&&d.send(JSON.stringify(e))}function C(){f=!1,r={...r,enabled:!1,state:"off"},p()}function K(){if(m)return;if(m=!0,d){try{d.onclose=null,d.close()}catch{}d=null}const e=y(),i=`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/session?token=${encodeURIComponent(e)}`;d=new WebSocket(i),d.onopen=()=>{m=!1,f=!0,p()},d.onclose=()=>{m=!1,C(),setTimeout(()=>{L()},1500)},d.onerror=()=>{m=!1,C()},d.onmessage=a=>{try{H(JSON.parse(a.data))}catch{}}}async function L(){T=await I(),p(),K()}p();L();
