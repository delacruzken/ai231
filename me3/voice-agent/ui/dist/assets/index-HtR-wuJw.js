(function(){const n=document.createElement("link").relList;if(n&&n.supports&&n.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))s(i);new MutationObserver(i=>{for(const r of i)if(r.type==="childList")for(const p of r.addedNodes)p.tagName==="LINK"&&p.rel==="modulepreload"&&s(p)}).observe(document,{childList:!0,subtree:!0});function t(i){const r={};return i.integrity&&(r.integrity=i.integrity),i.referrerPolicy&&(r.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?r.credentials="include":i.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function s(i){if(i.ep)return;i.ep=!0;const r=t(i);fetch(i.href,r)}})();const K="me3_ui_token";function ee(){return localStorage.getItem(K)||""}function P(e){localStorage.setItem(K,e)}function oe(){localStorage.removeItem(K)}async function ie(){const e=new URLSearchParams(location.search).get("token");if(e)return P(e),e;try{const n=await fetch("/api/ui-bootstrap");if(n.ok){const t=await n.json(),s=(t.token||"").trim();if(s)return P(s),s;if(!t.require_token)return oe(),""}}catch{}return ee()}let U="";function x(){return U||ee()}const re=document.querySelector("#app");let a={state:"off",enabled:!1,readiness:{audio:"degraded",vad:"degraded",asr:"degraded",tts:"degraded"},degraded:!0,transcript:[],mic_muted:!1},y=!1,u=null,L=!1,R=[],M=[],I=[],te="",k=!1,w=[],ae=0,O=null,m=null;const S=new Map;let _=null;const le=18,ce=2;let j=0,q=null;async function $(){const e=x();if(e)try{const[n,t,s]=await Promise.all([fetch("/api/timers",{headers:{Authorization:`Bearer ${e}`}}),fetch("/api/notes",{headers:{Authorization:`Bearer ${e}`}}),fetch("/api/jobs",{headers:{Authorization:`Bearer ${e}`}})]);if(n.ok&&(R=(await n.json()).timers||[]),t.ok&&(M=(await t.json()).notes||[]),s.ok){const i=await s.json();te=i.provider||"",I=i.jobs||[]}l()}catch{}}function E(e){const n=[];for(const t of e){const s=t.role||"user",i=t.type==="partial"||t.partial===!0,r=n[n.length-1];if(!!r&&(r.type==="partial"||r.partial===!0)&&(r.role||"user")===s){const f=r.ts||t.ts;n[n.length-1]=i?{...t,role:s,ts:f}:{...t,role:s,type:"final",partial:!1,final:!0,ts:f}}else r&&!i&&(r.role||"user")===s&&!(r.type==="partial"||r.partial===!0)&&G(r.text)===G(t.text)?n[n.length-1]={...r,...t,role:s,type:"final",partial:!1,final:!0,ts:r.ts||t.ts}:n.push(i?{...t,role:s}:{...t,role:s,type:t.type||"final",final:!0})}return n}function J(e){if(!e)return"";const n=new Date(e);return Number.isNaN(n.getTime())?"":n.toLocaleTimeString([],{hour:"numeric",minute:"2-digit",second:"2-digit"})}function de(e){const n=["Conversation"];for(const t of E(e)){const s=(t.text||"").trim();if(!s)continue;const i=(t.role||"user")==="assistant"?"Kaki":"You",r=J(t.ts);n.push(i),r&&n.push(r),n.push(s)}return n.join(`
`)}async function ue(){const e=de(a.transcript||[]);if(e==="Conversation")return!1;try{return await navigator.clipboard.writeText(e),!0}catch{const n=document.createElement("textarea");n.value=e,n.setAttribute("readonly",""),n.style.position="fixed",n.style.left="-9999px",document.body.appendChild(n),n.select();let t=!1;try{t=document.execCommand("copy")}finally{document.body.removeChild(n)}return t}}function G(e){return String(e||"").toLowerCase().replace(/[^a-z0-9\s]/g,"").trim()}function N(e,n="info",t){const s=++ae;w=[...w.slice(-4),{id:s,text:e,tone:n,jobId:t}],t&&(O=t,m!=null&&window.clearTimeout(m),m=window.setTimeout(()=>{O=null,m=null,l()},8e3)),l(),window.setTimeout(()=>{w=w.filter(i=>i.id!==s),l()},9e3)}function pe(e){a.transcript=E([...a.transcript||[],e])}function fe(e){const n=[];let t=[];for(const s of e)(s.role||"user")==="user"&&t.length&&(n.push(t),t=[]),t.push(s);return t.length&&n.push(t),n}function me(e){const n=fe(e),t=[];return n.forEach((s,i)=>{for(const r of s){const p=(r.role||"user")==="assistant",f=i*10+(p?2:1);t.push({row:r,score:f})}}),t.sort((s,i)=>i.score-s.score),t.map(s=>s.row)}function ge(){const e=document.querySelector("#transcript");if(!(e instanceof HTMLElement))return;const n=()=>{e.scrollTop=0};n(),requestAnimationFrame(()=>{n(),requestAnimationFrame(n)})}function he(e){return String(e.turn_id||e.ts||e.segment_id||e.text||"assistant").slice(0,96)}function ve(e,n,t){const s=S.get(e);if(!s){const i=!t||a.state==="speaking"||a.state==="thinking"||k;S.set(e,{target:n,shownLen:i?0:n.length,final:t}),i&&Q();return}n.startsWith(s.target.slice(0,Math.min(s.shownLen,s.target.length)))?s.target=n:!n.startsWith(s.target)&&!s.target.startsWith(n)?(s.target=n,s.shownLen=0):(s.target=n,s.shownLen=Math.min(s.shownLen,n.length)),s.final=t,s.shownLen<s.target.length&&Q()}function ye(e){for(const n of[...S.keys()])e.has(n)||S.delete(n)}function Q(){_==null&&(_=window.setInterval(()=>{let e=!1;for(const t of S.values())t.shownLen<t.target.length&&(t.shownLen=Math.min(t.target.length,t.shownLen+ce),e=!0);const n=[...S.values()].some(t=>t.shownLen<t.target.length);e&&l(),!n&&_!=null&&(window.clearInterval(_),_=null,e||l())},le))}function X(e,n,t){const s=e.role||"user",i=e.type==="final"||e.final===!0?"final":e.type==="partial"||e.partial===!0?"partial":"final",r=s==="assistant"?"Kaki":"You",p=n??(e.text||""),f=t?'<span class="caret" aria-hidden="true"></span>':"",A=!t&&i==="partial"?'<span class="live" aria-hidden="true"></span>':"",g=J(e.ts),T=g?`<time class="ts" datetime="${b(e.ts||"")}">${d(g)}</time>`:"";return`<div class="turn ${i} role-${b(s)}${t?" typing":""}">
    <div class="meta"><span class="who">${r}</span>${T}</div>
    <div class="bubble">${d(p)}${f}${A}</div>
  </div>`}function be(){if(a.state!=="thinking"&&!k)return!1;const e=E(a.transcript||[]);let n=-1;for(let t=e.length-1;t>=0;t--)if((e[t].role||"user")==="user"){n=t;break}if(n>=0){for(let t=n+1;t<e.length;t++)if((e[t].role||"")==="assistant"&&(e[t].text||"").trim())return!1}return!0}function we(){return a.state==="thinking"?"Thinking…":"Working…"}function l(){var z,D,H,B,F,Y,V;const e=Object.entries(a.readiness||{}).map(([o,c])=>`<span class="badge ${c}">${o}: ${c}</span>`).join(""),n=R.length===0?'<div class="empty">No active timers</div>':R.map(o=>`<div class="task-row"><span class="who">timer</span> ${d(o.label||o.id.slice(0,8))} · ${d(o.deadline_at)}</div>`).join(""),t=M.length===0?'<div class="empty">No notes yet</div>':M.slice(0,8).map(o=>`<div class="task-row"><span class="who">note</span> ${d((o.title?o.title+": ":"")+(o.body||""))}</div>`).join(""),s=me(E(a.transcript||[])),i=be(),r=new Set,p=i?`<div class="turn pending role-assistant" aria-live="polite" aria-busy="true">
         <div class="meta"><span class="who">Kaki</span><time class="ts">${d(J(new Date().toISOString()))}</time></div>
         <div class="bubble pending-bubble">
           <span class="spinner" aria-hidden="true"></span>
           <span class="pending-text">${d(we())}</span>
         </div>
       </div>`:"",f=s.map(o=>{if((o.role||"user")!=="assistant")return X(o);const c=he(o);r.add(c);const h=!(o.type==="partial"||o.partial===!0);ve(c,o.text||"",h);const v=S.get(c),se=v.shownLen<v.target.length||!v.final;return X(o,v.target.slice(0,v.shownLen),se)}).join("");ye(r);const A=s.length===0&&!i?'<div class="empty">No transcript yet. Turn ON and speak, or Send text turn.</div>':`${p}${f}`,g=!!a.mic_muted,T=!!a.enabled&&y,ne=w.length===0?"":`<div class="toast-stack" aria-live="polite">${w.map(o=>`<div class="toast tone-${b(o.tone)}" role="status">${d(o.text)}</div>`).join("")}</div>`;re.innerHTML=`
    <header class="top">
      <div>
        <h1 class="brand">ME3</h1>
        <p class="tagline">Voice session · tools · jobs</p>
      </div>
      <p class="conn ${y?"":"lost"}" id="conn">
        ${y?"Connected":"Connecting…"}
      </p>
    </header>
    ${ne}

    <div class="layout">
      <aside class="col col-controls">
        <div class="toggle-wrap">
          <button type="button" class="toggle ${a.enabled?"on":"off"}" id="toggle" ${y?"":"disabled"}>
            ${a.enabled?"ON":"OFF"}
          </button>
          <div class="status-line state-${b(a.state)}${i?" pending":""}">status <strong>${d(a.state)}</strong>${i?'<span class="spinner inline" aria-hidden="true"></span>':""}</div>
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
          <button type="button" id="refresh" ${y?"":"disabled"}>Refresh</button>
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
          <h2>Jobs <span class="who">${d(te||"—")}</span></h2>
          ${I.length===0?'<div class="empty">No jobs yet</div>':I.slice(0,6).map(o=>{const c=o.status==="needs_input"||o.status==="needs_approval",h=O===o.id?" flash":"";return`<div class="job-card status-${b(o.status)}${h}">
                <div><span class="who">${d(o.status)}</span> ${d((o.specification||o.id).slice(0,80))}</div>
                ${o.pending_user_request?`<div class="job-ask">${d(o.pending_user_request)}</div>`:""}
                ${o.outcome?`<div class="empty">${d(String(o.outcome).slice(0,120))}</div>`:""}
                <div class="job-actions">
                  ${c?`<button type="button" data-job-approve="${b(o.id)}">Approve / reply</button>`:""}
                  ${o.status==="queued"||o.status==="running"||o.status==="needs_input"||o.status==="needs_approval"?`<button type="button" data-job-cancel="${b(o.id)}">Cancel</button>`:""}
                </div>
              </div>`}).join("")}
        </div>
      </aside>

      <main class="col col-conversation">
        <section class="transcript-panel">
          <div class="panel-heading">
            <h2>Conversation</h2>
            <button
              type="button"
              class="copy-log ${Date.now()<j?"copied":""}"
              id="copy-log"
              ${s.length===0?"disabled":""}
              title="Copy conversation log (oldest → newest)"
            >${Date.now()<j?"Copied":"Copy log"}</button>
          </div>
          <div class="transcript" id="transcript">${A}</div>
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
  `,(z=document.querySelector("#save-token"))==null||z.addEventListener("click",()=>{const o=document.querySelector("#token").value.trim();o&&(U=o,P(o)),W()}),(D=document.querySelector("#toggle"))==null||D.addEventListener("click",()=>{C({type:"set_enabled",enabled:!a.enabled})}),(H=document.querySelector("#mic-mute"))==null||H.addEventListener("click",()=>{C({type:"set_mic_muted",muted:!a.mic_muted})}),(B=document.querySelector("#probe"))==null||B.addEventListener("click",()=>{C({type:"probe",text:"Session is on."})}),(F=document.querySelector("#say"))==null||F.addEventListener("click",async()=>{const o=window.prompt("User text turn");if(!o)return;k=!0,l();const c=x();try{await fetch("/api/say",{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${c}`},body:JSON.stringify({text:o})})}catch{k=!1,l()}}),(Y=document.querySelector("#refresh"))==null||Y.addEventListener("click",()=>{C({type:"get_session"}),$()}),(V=document.querySelector("#copy-log"))==null||V.addEventListener("click",()=>{ue().then(o=>{if(!o){N("Could not copy conversation log","warn");return}j=Date.now()+1600,q!=null&&window.clearTimeout(q),q=window.setTimeout(()=>{j=0,q=null,l()},1600),N("Conversation log copied","ok")})}),document.querySelectorAll("[data-job-approve]").forEach(o=>{o.addEventListener("click",async()=>{const c=o.dataset.jobApprove;if(!c)return;const h=window.prompt("Reply / approval note","approved")||"approved",v=x();await fetch(`/api/jobs/${c}/respond`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${v}`},body:JSON.stringify({response:h,approve:!0})}),$()})}),document.querySelectorAll("[data-job-cancel]").forEach(o=>{o.addEventListener("click",async()=>{const c=o.dataset.jobCancel;if(!c)return;const h=x();await fetch(`/api/jobs/${c}/cancel`,{method:"POST",headers:{Authorization:`Bearer ${h}`}}),$()})}),ge()}function d(e){return e.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}function b(e){return d(e).replace(/'/g,"&#39;")}function $e(e){const n=e.state||"off";(n==="thinking"||n==="speaking"||n==="off")&&(k=!1),a={state:n,enabled:!!e.enabled,readiness:e.readiness||a.readiness,degraded:!!e.degraded,transcript:E(e.transcript||a.transcript),last_error:e.last_error,utterance_generation:e.utterance_generation,mic_muted:!!e.mic_muted},l()}function ke(e){var n;if(e.type==="session"||e.state)$e(e),$();else if(e.type==="transcript")(e.role||"")==="assistant"&&(e.text||"").trim()&&(k=!1),pe(e),l();else if(e.type==="turn_done"||e.type==="turn_cancelled"||e.type==="error")k=!1,e.type==="error"&&e.message&&(a.last_error=String(e.message)),l();else if(e.type==="readiness")a.readiness=e.readiness||a.readiness,l();else if(e.type==="notification"){const t=String(((n=e.payload)==null?void 0:n.message)||e.message||"").trim();if(t){const s=String(e.status||"");N(t,s==="failed"?"warn":s==="completed"?"ok":"info",e.job_id?String(e.job_id):void 0)}$()}else if(e.type==="notification_delivered"||e.type==="tool_result")$();else if(e.type==="job_update"){const t=e.job||{},s=String(t.status||"");if((s==="completed"||s==="failed")&&!w.some(i=>i.jobId&&t.id&&i.jobId===String(t.id))){const i=String(t.specification||t.id||"job").split(`
`)[0].slice(0,72);N(s==="completed"?`Job finished: ${i}`:`Job failed: ${i}`,s==="failed"?"warn":"ok",t.id?String(t.id):void 0)}else t.id&&(s==="completed"||s==="failed")&&(O=String(t.id),m!=null&&window.clearTimeout(m),m=window.setTimeout(()=>{O=null,m=null,l()},8e3));$()}}function C(e){u&&u.readyState===WebSocket.OPEN&&u.send(JSON.stringify(e))}function Z(){y=!1,a={...a,enabled:!1,state:"off"},l()}function Se(){if(L)return;if(L=!0,u){try{u.onclose=null,u.close()}catch{}u=null}const e=x(),t=`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/session?token=${encodeURIComponent(e)}`;u=new WebSocket(t),u.onopen=()=>{L=!1,y=!0,l()},u.onclose=()=>{L=!1,Z(),setTimeout(()=>{W()},1500)},u.onerror=()=>{L=!1,Z()},u.onmessage=s=>{try{ke(JSON.parse(s.data))}catch{}}}async function W(){U=await ie(),l(),Se()}l();W();
