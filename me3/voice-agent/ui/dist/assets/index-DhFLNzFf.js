(function(){const n=document.createElement("link").relList;if(n&&n.supports&&n.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))s(o);new MutationObserver(o=>{for(const r of o)if(r.type==="childList")for(const p of r.addedNodes)p.tagName==="LINK"&&p.rel==="modulepreload"&&s(p)}).observe(document,{childList:!0,subtree:!0});function t(o){const r={};return o.integrity&&(r.integrity=o.integrity),o.referrerPolicy&&(r.referrerPolicy=o.referrerPolicy),o.crossOrigin==="use-credentials"?r.credentials="include":o.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function s(o){if(o.ep)return;o.ep=!0;const r=t(o);fetch(o.href,r)}})();const M="me3_ui_token";function G(){return localStorage.getItem(M)||""}function P(e){localStorage.setItem(M,e)}function te(){localStorage.removeItem(M)}async function ne(){const e=new URLSearchParams(location.search).get("token");if(e)return P(e),e;try{const n=await fetch("/api/ui-bootstrap");if(n.ok){const t=await n.json(),s=(t.token||"").trim();if(s)return P(s),s;if(!t.require_token)return te(),""}}catch{}return G()}let I="";function O(){return I||G()}const se=document.querySelector("#app");let a={state:"off",enabled:!1,readiness:{audio:"degraded",vad:"degraded",asr:"degraded",tts:"degraded"},degraded:!0,transcript:[],mic_muted:!1},b=!1,u=null,_=!1,x=[],A=[],R=[],Q="",k=!1,w=[],ie=0,j=null,m=null;const S=new Map;let L=null;const oe=18,re=2;async function $(){const e=O();if(e)try{const[n,t,s]=await Promise.all([fetch("/api/timers",{headers:{Authorization:`Bearer ${e}`}}),fetch("/api/notes",{headers:{Authorization:`Bearer ${e}`}}),fetch("/api/jobs",{headers:{Authorization:`Bearer ${e}`}})]);if(n.ok&&(x=(await n.json()).timers||[]),t.ok&&(A=(await t.json()).notes||[]),s.ok){const o=await s.json();Q=o.provider||"",R=o.jobs||[]}l()}catch{}}function E(e){const n=[];for(const t of e){const s=t.role||"user",o=t.type==="partial"||t.partial===!0,r=n[n.length-1];if(!!r&&(r.type==="partial"||r.partial===!0)&&(r.role||"user")===s){const f=r.ts||t.ts;n[n.length-1]=o?{...t,role:s,ts:f}:{...t,role:s,type:"final",partial:!1,final:!0,ts:f}}else r&&!o&&(r.role||"user")===s&&!(r.type==="partial"||r.partial===!0)&&B(r.text)===B(t.text)?n[n.length-1]={...r,...t,role:s,type:"final",partial:!1,final:!0,ts:r.ts||t.ts}:n.push(o?{...t,role:s}:{...t,role:s,type:t.type||"final",final:!0})}return n}function X(e){if(!e)return"";const n=new Date(e);return Number.isNaN(n.getTime())?"":n.toLocaleTimeString([],{hour:"numeric",minute:"2-digit",second:"2-digit"})}function B(e){return String(e||"").toLowerCase().replace(/[^a-z0-9\s]/g,"").trim()}function D(e,n="info",t){const s=++ie;w=[...w.slice(-4),{id:s,text:e,tone:n,jobId:t}],t&&(j=t,m!=null&&window.clearTimeout(m),m=window.setTimeout(()=>{j=null,m=null,l()},8e3)),l(),window.setTimeout(()=>{w=w.filter(o=>o.id!==s),l()},9e3)}function ae(e){a.transcript=E([...a.transcript||[],e])}function le(e){const n=[];let t=[];for(const s of e)(s.role||"user")==="user"&&t.length&&(n.push(t),t=[]),t.push(s);return t.length&&n.push(t),n}function ce(e){const n=le(e),t=[];return n.forEach((s,o)=>{for(const r of s){const p=(r.role||"user")==="assistant",f=o*10+(p?2:1);t.push({row:r,score:f})}}),t.sort((s,o)=>o.score-s.score),t.map(s=>s.row)}function de(){const e=document.querySelector("#transcript");if(!(e instanceof HTMLElement))return;const n=()=>{e.scrollTop=0};n(),requestAnimationFrame(()=>{n(),requestAnimationFrame(n)})}function ue(e){return String(e.turn_id||e.ts||e.segment_id||e.text||"assistant").slice(0,96)}function pe(e,n,t){const s=S.get(e);if(!s){const o=!t||a.state==="speaking"||a.state==="thinking"||k;S.set(e,{target:n,shownLen:o?0:n.length,final:t}),o&&F();return}n.startsWith(s.target.slice(0,Math.min(s.shownLen,s.target.length)))?s.target=n:!n.startsWith(s.target)&&!s.target.startsWith(n)?(s.target=n,s.shownLen=0):(s.target=n,s.shownLen=Math.min(s.shownLen,n.length)),s.final=t,s.shownLen<s.target.length&&F()}function fe(e){for(const n of[...S.keys()])e.has(n)||S.delete(n)}function F(){L==null&&(L=window.setInterval(()=>{let e=!1;for(const t of S.values())t.shownLen<t.target.length&&(t.shownLen=Math.min(t.target.length,t.shownLen+re),e=!0);const n=[...S.values()].some(t=>t.shownLen<t.target.length);e&&l(),!n&&L!=null&&(window.clearInterval(L),L=null,e||l())},oe))}function Y(e,n,t){const s=e.role||"user",o=e.type==="final"||e.final===!0?"final":e.type==="partial"||e.partial===!0?"partial":"final",r=s==="assistant"?"Kaki":"You",p=n??(e.text||""),f=t?'<span class="caret" aria-hidden="true"></span>':"",N=!t&&o==="partial"?'<span class="live" aria-hidden="true"></span>':"",g=X(e.ts),T=g?`<time class="ts" datetime="${y(e.ts||"")}">${d(g)}</time>`:"";return`<div class="turn ${o} role-${y(s)}${t?" typing":""}">
    <div class="meta"><span class="who">${r}</span>${T}</div>
    <div class="bubble">${d(p)}${f}${N}</div>
  </div>`}function me(){if(a.state!=="thinking"&&!k)return!1;const e=E(a.transcript||[]);let n=-1;for(let t=e.length-1;t>=0;t--)if((e[t].role||"user")==="user"){n=t;break}if(n>=0){for(let t=n+1;t<e.length;t++)if((e[t].role||"")==="assistant"&&(e[t].text||"").trim())return!1}return!0}function ge(){return a.state==="thinking"?"Thinking…":"Working…"}function l(){var J,K,U,W,z,H;const e=Object.entries(a.readiness||{}).map(([i,c])=>`<span class="badge ${c}">${i}: ${c}</span>`).join(""),n=x.length===0?'<div class="empty">No active timers</div>':x.map(i=>`<div class="task-row"><span class="who">timer</span> ${d(i.label||i.id.slice(0,8))} · ${d(i.deadline_at)}</div>`).join(""),t=A.length===0?'<div class="empty">No notes yet</div>':A.slice(0,8).map(i=>`<div class="task-row"><span class="who">note</span> ${d((i.title?i.title+": ":"")+(i.body||""))}</div>`).join(""),s=ce(E(a.transcript||[])),o=me(),r=new Set,p=o?`<div class="turn pending role-assistant" aria-live="polite" aria-busy="true">
         <div class="meta"><span class="who">Kaki</span><time class="ts">${d(X(new Date().toISOString()))}</time></div>
         <div class="bubble pending-bubble">
           <span class="spinner" aria-hidden="true"></span>
           <span class="pending-text">${d(ge())}</span>
         </div>
       </div>`:"",f=s.map(i=>{if((i.role||"user")!=="assistant")return Y(i);const c=ue(i);r.add(c);const h=!(i.type==="partial"||i.partial===!0);pe(c,i.text||"",h);const v=S.get(c),ee=v.shownLen<v.target.length||!v.final;return Y(i,v.target.slice(0,v.shownLen),ee)}).join("");fe(r);const N=s.length===0&&!o?'<div class="empty">No transcript yet. Turn ON and speak, or Send text turn.</div>':`${p}${f}`,g=!!a.mic_muted,T=!!a.enabled&&b,Z=w.length===0?"":`<div class="toast-stack" aria-live="polite">${w.map(i=>`<div class="toast tone-${y(i.tone)}" role="status">${d(i.text)}</div>`).join("")}</div>`;se.innerHTML=`
    <header class="top">
      <div>
        <h1 class="brand">ME3</h1>
        <p class="tagline">Voice session · tools · jobs</p>
      </div>
      <p class="conn ${b?"":"lost"}" id="conn">
        ${b?"Connected":"Connecting…"}
      </p>
    </header>
    ${Z}

    <div class="layout">
      <aside class="col col-controls">
        <div class="toggle-wrap">
          <button type="button" class="toggle ${a.enabled?"on":"off"}" id="toggle" ${b?"":"disabled"}>
            ${a.enabled?"ON":"OFF"}
          </button>
          <div class="status-line state-${y(a.state)}${o?" pending":""}">status <strong>${d(a.state)}</strong>${o?'<span class="spinner inline" aria-hidden="true"></span>':""}</div>
        </div>

        <button
          type="button"
          class="mic-btn ${g?"muted":"live"}"
          id="mic-mute"
          ${T?"":"disabled"}
          title="${g?"Unmute microphone":"Mute microphone (session stays on)"}"
        >
          ${g?"Mic muted":"Mic open"}
        </button>
        <p class="mic-hint">${g?"Mic muted for new questions — you can still barge in to stop Kaki mid-speech. Unmute to ask.":"Listening while ON. Speak over Kaki to interrupt. Headphones improve ASR + barge-in."}</p>

        <div class="badges">${e}</div>

        <div class="actions">
          <button type="button" id="probe" ${T?"":"disabled"}>Probe TTS</button>
          <button type="button" id="say" ${T?"":"disabled"}>Send text turn</button>
          <button type="button" id="refresh" ${b?"":"disabled"}>Refresh</button>
        </div>

        <div class="side-panel">
          <h2>Timers</h2>
          ${n}
        </div>
        <div class="side-panel">
          <h2>Notes</h2>
          ${t}
        </div>
        <div class="side-panel jobs-panel">
          <h2>Jobs <span class="who">${d(Q||"—")}</span></h2>
          ${R.length===0?'<div class="empty">No jobs yet</div>':R.slice(0,6).map(i=>{const c=i.status==="needs_input"||i.status==="needs_approval",h=j===i.id?" flash":"";return`<div class="job-card status-${y(i.status)}${h}">
                <div><span class="who">${d(i.status)}</span> ${d((i.specification||i.id).slice(0,80))}</div>
                ${i.pending_user_request?`<div class="job-ask">${d(i.pending_user_request)}</div>`:""}
                ${i.outcome?`<div class="empty">${d(String(i.outcome).slice(0,120))}</div>`:""}
                <div class="job-actions">
                  ${c?`<button type="button" data-job-approve="${y(i.id)}">Approve / reply</button>`:""}
                  ${i.status==="queued"||i.status==="running"||i.status==="needs_input"||i.status==="needs_approval"?`<button type="button" data-job-cancel="${y(i.id)}">Cancel</button>`:""}
                </div>
              </div>`}).join("")}
        </div>
      </aside>

      <main class="col col-conversation">
        <section class="transcript-panel">
          <h2>Conversation</h2>
          <div class="transcript" id="transcript">${N}</div>
        </section>
        <details class="diagnostics">
          <summary>Diagnostics</summary>
          <div class="token-bar">
            <input id="token" type="password" placeholder="Override UI token (optional)" value="" />
            <button type="button" id="save-token">Save override</button>
          </div>
          <pre>${d(JSON.stringify(a,null,2))}</pre>
        </details>
      </main>
    </div>
  `,(J=document.querySelector("#save-token"))==null||J.addEventListener("click",()=>{const i=document.querySelector("#token").value.trim();i&&(I=i,P(i)),C()}),(K=document.querySelector("#toggle"))==null||K.addEventListener("click",()=>{q({type:"set_enabled",enabled:!a.enabled})}),(U=document.querySelector("#mic-mute"))==null||U.addEventListener("click",()=>{q({type:"set_mic_muted",muted:!a.mic_muted})}),(W=document.querySelector("#probe"))==null||W.addEventListener("click",()=>{q({type:"probe",text:"Session is on."})}),(z=document.querySelector("#say"))==null||z.addEventListener("click",async()=>{const i=window.prompt("User text turn");if(!i)return;k=!0,l();const c=O();try{await fetch("/api/say",{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${c}`},body:JSON.stringify({text:i})})}catch{k=!1,l()}}),(H=document.querySelector("#refresh"))==null||H.addEventListener("click",()=>{q({type:"get_session"}),$()}),document.querySelectorAll("[data-job-approve]").forEach(i=>{i.addEventListener("click",async()=>{const c=i.dataset.jobApprove;if(!c)return;const h=window.prompt("Reply / approval note","approved")||"approved",v=O();await fetch(`/api/jobs/${c}/respond`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${v}`},body:JSON.stringify({response:h,approve:!0})}),$()})}),document.querySelectorAll("[data-job-cancel]").forEach(i=>{i.addEventListener("click",async()=>{const c=i.dataset.jobCancel;if(!c)return;const h=O();await fetch(`/api/jobs/${c}/cancel`,{method:"POST",headers:{Authorization:`Bearer ${h}`}}),$()})}),de()}function d(e){return e.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}function y(e){return d(e).replace(/'/g,"&#39;")}function he(e){const n=e.state||"off";(n==="thinking"||n==="speaking"||n==="off")&&(k=!1),a={state:n,enabled:!!e.enabled,readiness:e.readiness||a.readiness,degraded:!!e.degraded,transcript:E(e.transcript||a.transcript),last_error:e.last_error,utterance_generation:e.utterance_generation,mic_muted:!!e.mic_muted},l()}function ve(e){var n;if(e.type==="session"||e.state)he(e),$();else if(e.type==="transcript")(e.role||"")==="assistant"&&(e.text||"").trim()&&(k=!1),ae(e),l();else if(e.type==="turn_done"||e.type==="turn_cancelled"||e.type==="error")k=!1,e.type==="error"&&e.message&&(a.last_error=String(e.message)),l();else if(e.type==="readiness")a.readiness=e.readiness||a.readiness,l();else if(e.type==="notification"){const t=String(((n=e.payload)==null?void 0:n.message)||e.message||"").trim();if(t){const s=String(e.status||"");D(t,s==="failed"?"warn":s==="completed"?"ok":"info",e.job_id?String(e.job_id):void 0)}$()}else if(e.type==="notification_delivered"||e.type==="tool_result")$();else if(e.type==="job_update"){const t=e.job||{},s=String(t.status||"");if((s==="completed"||s==="failed")&&!w.some(o=>o.jobId&&t.id&&o.jobId===String(t.id))){const o=String(t.specification||t.id||"job").split(`
`)[0].slice(0,72);D(s==="completed"?`Job finished: ${o}`:`Job failed: ${o}`,s==="failed"?"warn":"ok",t.id?String(t.id):void 0)}else t.id&&(s==="completed"||s==="failed")&&(j=String(t.id),m!=null&&window.clearTimeout(m),m=window.setTimeout(()=>{j=null,m=null,l()},8e3));$()}}function q(e){u&&u.readyState===WebSocket.OPEN&&u.send(JSON.stringify(e))}function V(){b=!1,a={...a,enabled:!1,state:"off"},l()}function be(){if(_)return;if(_=!0,u){try{u.onclose=null,u.close()}catch{}u=null}const e=O(),t=`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/session?token=${encodeURIComponent(e)}`;u=new WebSocket(t),u.onopen=()=>{_=!1,b=!0,l()},u.onclose=()=>{_=!1,V(),setTimeout(()=>{C()},1500)},u.onerror=()=>{_=!1,V()},u.onmessage=s=>{try{ve(JSON.parse(s.data))}catch{}}}async function C(){I=await ne(),l(),be()}l();C();
