(function(){const s=document.createElement("link").relList;if(s&&s.supports&&s.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))t(i);new MutationObserver(i=>{for(const o of i)if(o.type==="childList")for(const a of o.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&t(a)}).observe(document,{childList:!0,subtree:!0});function n(i){const o={};return i.integrity&&(o.integrity=i.integrity),i.referrerPolicy&&(o.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?o.credentials="include":i.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function t(i){if(i.ep)return;i.ep=!0;const o=n(i);fetch(i.href,o)}})();const V="me3_ui_token",F=new Set(["queued","running","needs_input","needs_approval"]);function ue(){return localStorage.getItem(V)||""}function W(e){localStorage.setItem(V,e)}function ge(){localStorage.removeItem(V)}async function ve(){const e=new URLSearchParams(location.search).get("token");if(e)return W(e),e;try{const s=await fetch("/api/ui-bootstrap");if(s.ok){const n=await s.json(),t=(n.token||"").trim();if(t)return W(t),t;if(!n.require_token)return ge(),""}}catch{}return ue()}let G="";function Q(){return G||ue()}const be=document.querySelector("#app");let l={state:"off",enabled:!1,readiness:{audio:"degraded",vad:"degraded",asr:"degraded",tts:"degraded"},degraded:!0,transcript:[],mic_muted:!1},S=!1,h=null,C=!1,N=[],Y=[],m=[],de="",L=!1,T=[],ye=0,z=null,I=null;const E=new Map;let q=null;const $e=18,we=2;let K=0,D=null,A={},_=null,g="",R=null,k=null,O=null;function ke(){const e=document.activeElement;return!(e instanceof HTMLInputElement||e instanceof HTMLTextAreaElement)||!e.id?null:{id:e.id,start:e.selectionStart??0,end:e.selectionEnd??0}}function Se(e){if(!e)return;const s=document.getElementById(e.id);if(s instanceof HTMLInputElement||s instanceof HTMLTextAreaElement){s.focus();try{s.setSelectionRange(e.start,e.end)}catch{}}}function X(e=!1){const s={Authorization:`Bearer ${Q()}`};return e&&(s["Content-Type"]="application/json"),s}async function fe(e){var s,n;try{const t=await e.json();if(typeof(t==null?void 0:t.detail)=="string")return t.detail;if((s=t==null?void 0:t.detail)!=null&&s.message)return String(t.detail.message);if((n=t==null?void 0:t.error)!=null&&n.message)return String(t.error.message)}catch{}return`Request failed (${e.status})`}function Z(e){if(!(e!=null&&e.id))return;const s=m.findIndex(n=>n.id===e.id);s>=0?m=[...m.slice(0,s),{...m[s],...e},...m.slice(s+1)]:m=[e,...m].slice(0,20),pe()}function U(){return m.some(e=>F.has(e.status))}function pe(){if(U()){if(k!=null)return;k=window.setInterval(()=>{y(),!U()&&k!=null&&(window.clearInterval(k),k=null)},4e3)}else k!=null&&(window.clearInterval(k),k=null)}function me(e){e&&(z=e,I!=null&&window.clearTimeout(I),I=window.setTimeout(()=>{z=null,I=null,c()},8e3))}function ne(e){const s=(e.specification||e.id||"job").trim().split(`
`)[0];if(!s)return e.id.slice(0,8);if(s.length<=160)return s;const n=s.slice(0,160),t=n.lastIndexOf(" ");return`${t>100?n.slice(0,t):n}…`}function Te(e,s){const n=String(e||"").trim();if(!n)return"";if(/^\s*[\[{]/.test(n)||n.includes("{'type'")||n.includes('"type": "text"'))return s==="completed"?"":"Build failed (see logs).";const t=n.split(/\n/).map(r=>r.trim()).filter(Boolean),i=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\/?$/i,o=t.filter(r=>i.test(r)||/^\d+$/.test(r)&&r.length<=4).length;if(o>=2&&o>=Math.ceil(t.length/2))return s==="completed"?"":"Build failed (see logs).";const a=t.filter(r=>!i.test(r)&&!(/^\d+$/.test(r)&&r.length<=4)).join(" ").replace(/\s+/g," ").trim();return a.length<12||s==="completed"&&!/^build finished/i.test(a)?"":a.length>160?`${a.slice(0,160)}…`:a}function se(e){return e==="completed"?"done":e==="needs_input"?"needs input":e==="needs_approval"?"needs approval":e.replace(/_/g," ")}function _e(e){const s=P(e);if(!s)return"";const n=Math.max(0,Math.floor((Date.now()-s.getTime())/1e3));if(n<60)return`${n}s`;const t=Math.floor(n/60),i=n%60;return t<60?`${t}:${String(i).padStart(2,"0")}`:`${Math.floor(t/60)}h ${t%60}m`}function Le(e,s){const n=P(e),t=P(s);if(!n||!t)return"";const i=Math.max(0,Math.floor((t.getTime()-n.getTime())/1e3));if(i<60)return`${i}s`;const o=Math.floor(i/60),a=i%60;return`${o}:${String(a).padStart(2,"0")}`}function ie(e){const s=e.created_at||e.updated_at||e.updated_at_local;if(e.status==="queued"||e.status==="running"){const n=_e(s);return n?`↑ ${n}`:"live"}if(e.status==="needs_input"||e.status==="needs_approval")return"waiting on you";if(e.status==="completed"){const n=Le(e.created_at,e.updated_at||e.updated_at_local);return n?`took ${n}`:"done"}return e.status==="failed"?"failed":Ee(e.updated_at_local||e.updated_at||e.created_at)}function P(e){if(!e)return null;const s=new Date(e);return Number.isNaN(s.getTime())?null:s}function Ee(e){const s=P(e);if(!s)return"";const n=Math.round((Date.now()-s.getTime())/1e3);if(n<10)return"just now";if(n<60)return`${n}s ago`;const t=Math.round(n/60);if(t<60)return`${t}m ago`;const i=Math.round(t/60);return i<48?`${i}h ago`:s.toLocaleDateString([],{month:"short",day:"numeric"})}function Me(e){const s=P(e);if(!s)return e;const n=Math.round((s.getTime()-Date.now())/1e3);if(n<=0){const i=Math.abs(n);return i<60?"due now":i<3600?`${Math.round(i/60)}m overdue`:`${Math.round(i/3600)}h overdue`}if(n<60)return`in ${n}s`;if(n<3600){const i=Math.floor(n/60),o=n%60;return o?`in ${i}m ${o}s`:`in ${i}m`}if(n<86400){const i=Math.floor(n/3600),o=Math.round(n%3600/60);return o?`in ${i}h ${o}m`:`in ${i}h`}return`in ${Math.round(n/86400)}d`}async function y(){const e=Q();if(e)try{const[s,n,t]=await Promise.all([fetch("/api/timers",{headers:{Authorization:`Bearer ${e}`}}),fetch("/api/notes",{headers:{Authorization:`Bearer ${e}`}}),fetch("/api/jobs",{headers:{Authorization:`Bearer ${e}`}})]);if(s.ok&&(N=(await s.json()).timers||[]),n.ok&&(Y=(await n.json()).notes||[]),t.ok){const i=await t.json();de=i.provider||"",m=i.jobs||[]}pe(),(N.length||U())&&We(),c()}catch{}}function J(e){const s=[];for(const n of e){const t=n.role||"user",i=n.type==="partial"||n.partial===!0,o=s[s.length-1];if(!!o&&(o.type==="partial"||o.partial===!0)&&(o.role||"user")===t){const r=o.ts||n.ts;s[s.length-1]=i?{...n,role:t,ts:r}:{...n,role:t,type:"final",partial:!1,final:!0,ts:r}}else o&&!i&&(o.role||"user")===t&&!(o.type==="partial"||o.partial===!0)&&oe(o.text)===oe(n.text)?s[s.length-1]={...o,...n,role:t,type:"final",partial:!1,final:!0,ts:o.ts||n.ts}:s.push(i?{...n,role:t}:{...n,role:t,type:n.type||"final",final:!0})}return s}function ee(e){if(!e)return"";const s=new Date(e);return Number.isNaN(s.getTime())?"":s.toLocaleTimeString([],{hour:"numeric",minute:"2-digit",second:"2-digit"})}function xe(e){const s=["Conversation"];for(const n of J(e)){const t=(n.text||"").trim();if(!t)continue;const i=(n.role||"user")==="assistant"?"Kaki":"You",o=ee(n.ts);s.push(i),o&&s.push(o),s.push(t)}return s.join(`
`)}async function Ce(){const e=xe(l.transcript||[]);if(e==="Conversation")return!1;try{return await navigator.clipboard.writeText(e),!0}catch{const s=document.createElement("textarea");s.value=e,s.setAttribute("readonly",""),s.style.position="fixed",s.style.left="-9999px",document.body.appendChild(s),s.select();let n=!1;try{n=document.execCommand("copy")}finally{document.body.removeChild(s)}return n}}function oe(e){return String(e||"").toLowerCase().replace(/[^a-z0-9\s]/g,"").trim()}function v(e,s="info",n){const t=++ye;T=[...T.slice(-4),{id:t,text:e,tone:s,jobId:n}],n&&me(n),c(),window.setTimeout(()=>{T=T.filter(i=>i.id!==t),c()},9e3)}function qe(e){l.transcript=J([...l.transcript||[],e])}function Oe(e){const s=[];let n=[];for(const t of e){const i=t.role||"user",o=(t.kind||"")==="notification"||(t.kind||"")==="clarify";(i==="user"&&!o||o)&&n.length&&(s.push(n),n=[]),n.push(t)}return n.length&&s.push(n),s}function Ae(e){const s=Oe(e),n=[];return s.forEach((t,i)=>{for(const o of t){const a=(o.role||"user")==="assistant",r=i*10+(a?2:1);n.push({row:o,score:r})}}),n.sort((t,i)=>i.score-t.score),n.map(t=>t.row)}function Ne(){const e=document.querySelector("#transcript");if(!(e instanceof HTMLElement))return;const s=()=>{e.scrollTop=0};s(),requestAnimationFrame(()=>{s(),requestAnimationFrame(s)})}function Re(e){return String(e.turn_id||e.ts||e.segment_id||e.text||"assistant").slice(0,96)}function Pe(e,s,n){const t=E.get(e);if(!t){const i=!n||l.state==="speaking"||l.state==="thinking"||L;E.set(e,{target:s,shownLen:i?0:s.length,final:n}),i&&re();return}s.startsWith(t.target.slice(0,Math.min(t.shownLen,t.target.length)))?t.target=s:!s.startsWith(t.target)&&!t.target.startsWith(s)?(t.target=s,t.shownLen=0):(t.target=s,t.shownLen=Math.min(t.shownLen,s.length)),t.final=n,t.shownLen<t.target.length&&re()}function Je(e){for(const s of[...E.keys()])e.has(s)||E.delete(s)}function re(){q==null&&(q=window.setInterval(()=>{let e=!1;for(const n of E.values())n.shownLen<n.target.length&&(n.shownLen=Math.min(n.target.length,n.shownLen+we),e=!0);const s=[...E.values()].some(n=>n.shownLen<n.target.length);e&&c(),!s&&q!=null&&(window.clearInterval(q),q=null,e||c())},$e))}function ae(e,s,n){const t=e.role||"user",i=e.type==="final"||e.final===!0?"final":e.type==="partial"||e.partial===!0?"partial":"final",o=t==="assistant"?"Kaki":"You",a=s??(e.text||""),r=n?'<span class="caret" aria-hidden="true"></span>':"",u=!n&&i==="partial"?'<span class="live" aria-hidden="true"></span>':"",b=ee(e.ts),$=b?`<time class="ts" datetime="${p(e.ts||"")}">${d(b)}</time>`:"";return`<div class="turn ${i} role-${p(t)}${n?" typing":""}">
    <div class="meta"><span class="who">${o}</span>${$}</div>
    <div class="bubble">${d(a)}${r}${u}</div>
  </div>`}function je(){if(l.state!=="thinking"&&!L)return!1;const e=J(l.transcript||[]);let s=-1;for(let n=e.length-1;n>=0;n--)if((e[n].role||"user")==="user"){s=n;break}if(s>=0){for(let n=s+1;n<e.length;n++)if((e[n].role||"")==="assistant"&&(e[n].text||"").trim())return!1}return!0}function Ie(){return l.state==="thinking"?"Thinking…":"Working…"}function De(){const e=m.length===0?'<div class="empty">No jobs yet — ask Kaki to start one.</div>':m.slice(0,8).map(t=>{const i=t.status==="needs_input"||t.status==="needs_approval",o=F.has(t.status),a=z===t.id?" flash":"",r=_===t.id,u=R===t.id,b=ie(t),$=t.status==="queued"||t.status==="running"?'<span class="job-live" aria-hidden="true"></span>':"",M=A[t.id]??"";return`<div class="job-card status-${p(t.status)}${a}" data-job-id="${p(t.id)}">
              <div class="job-card-top">
                <span class="status-pill status-${p(t.status)}">${$}${d(se(t.status))}</span>
                ${b?`<span class="job-when">${d(b)}</span>`:""}
              </div>
              <div class="job-title">${d(ne(t))}</div>
              ${t.pending_user_request?`<div class="job-ask">${d(t.pending_user_request)}</div>`:""}
              ${(()=>{if(!(t.status==="completed"||t.status==="failed"))return"";const j=Te(t.outcome,t.status);return j?`<div class="job-outcome">${d(j)}</div>`:""})()}
              ${i?`<div class="job-reply">
                      <textarea
                        id="job-reply-${p(t.id)}"
                        data-job-reply="${p(t.id)}"
                        rows="2"
                        placeholder="${t.status==="needs_approval"?"Optional note, or leave blank to approve":"Your reply…"}"
                        ${r?"disabled":""}
                      >${d(M)}</textarea>
                      <div class="job-actions">
                        <button type="button" class="btn-primary" data-job-reply-send="${p(t.id)}" ${r?"disabled":""}>${r?"Sending…":"Reply"}</button>
                        <button type="button" data-job-approve="${p(t.id)}" ${r?"disabled":""}>Approve</button>
                      </div>
                    </div>`:""}
              ${o?`<div class="job-actions job-actions-secondary">
                      ${u?`<span class="job-confirm-label">Cancel this job?</span>
                             <button type="button" class="btn-danger" data-job-cancel-yes="${p(t.id)}" ${r?"disabled":""}>${r?"Cancelling…":"Yes, cancel"}</button>
                             <button type="button" data-job-cancel-no="${p(t.id)}" ${r?"disabled":""}>Keep</button>`:`<button type="button" data-job-cancel="${p(t.id)}" ${r?"disabled":""}>Cancel</button>`}
                    </div>`:""}
            </div>`}).join(""),s=m.find(t=>F.has(t.status)),n=s?`<div class="jobs-live-banner status-${p(s.status)}" aria-live="polite">
         <span class="job-live" aria-hidden="true"></span>
         <strong>${d(se(s.status))}</strong>
         <span>${d(ie(s))}</span>
         <span class="jobs-live-title">${d(ne(s))}</span>
       </div>`:m.some(t=>t.status==="completed")?`<div class="jobs-live-banner status-completed" aria-live="polite">
           <strong>done</strong>
           <span>latest build finished</span>
         </div>`:"";return`<div class="side-panel jobs-panel">
    <h2>Jobs <span class="who">${d(de||"—")}</span></h2>
    ${n}
    ${g?`<div class="form-error" role="alert">${d(g)}</div>`:""}
    <div class="job-list">${e}</div>
  </div>`}function c(){const e=ke(),s=Object.entries(l.readiness||{}).map(([f,w])=>`<span class="badge ${w}">${f}: ${w}</span>`).join(""),n=N.length===0?'<div class="empty">No active timers</div>':N.map(f=>{const w=f.label||f.id.slice(0,8),B=Me(f.deadline_at);return`<div class="task-row"><span class="who">timer</span> ${d(w)} · <span class="timer-countdown">${d(B)}</span></div>`}).join(""),t=Y.length===0?'<div class="empty">No notes yet</div>':Y.slice(0,8).map(f=>`<div class="task-row"><span class="who">note</span> ${d((f.title?f.title+": ":"")+(f.body||""))}</div>`).join(""),i=Ae(J(l.transcript||[])),o=je(),a=new Set,r=o?`<div class="turn pending role-assistant" aria-live="polite" aria-busy="true">
         <div class="meta"><span class="who">Kaki</span><time class="ts">${d(ee(new Date().toISOString()))}</time></div>
         <div class="bubble pending-bubble">
           <span class="spinner" aria-hidden="true"></span>
           <span class="pending-text">${d(Ie())}</span>
         </div>
       </div>`:"",u=i.map(f=>{if((f.role||"user")!=="assistant")return ae(f);const w=Re(f);a.add(w);const B=!(f.type==="partial"||f.partial===!0);Pe(w,f.text||"",B);const x=E.get(w),he=x.shownLen<x.target.length||!x.final;return ae(f,x.target.slice(0,x.shownLen),he)}).join("");Je(a);const b=i.length===0&&!o?'<div class="empty">No transcript yet. Turn ON and speak, or Send text turn.</div>':`${r}${u}`,$=!!l.mic_muted,M=!!l.enabled&&S,j=T.length===0?"":`<div class="toast-stack" aria-live="polite">${T.map(f=>`<div class="toast tone-${p(f.tone)}" role="status">${d(f.text)}</div>`).join("")}</div>`;be.innerHTML=`
    <header class="top">
      <div>
        <h1 class="brand">Kaki</h1>
      </div>
      <p class="conn ${S?"":"lost"}" id="conn">
        ${S?"Connected":"Connecting…"}
      </p>
    </header>
    ${j}

    <div class="layout">
      <aside class="col col-controls">
        <div class="toggle-wrap">
          <button type="button" class="toggle ${l.enabled?"on":"off"}" id="toggle" ${S?"":"disabled"}>
            ${l.enabled?"ON":"OFF"}
          </button>
          <div class="status-line state-${p(l.state)}${o?" pending":""}">status <strong>${d(l.state)}</strong>${o?'<span class="spinner inline" aria-hidden="true"></span>':""}</div>
        </div>

        <button
          type="button"
          class="mic-btn ${$?"muted":"live"}"
          id="mic-mute"
          ${M?"":"disabled"}
          title="${$?"Unmute microphone":"Mute microphone (session stays on)"}"
        >
          ${$?"Mic muted":"Mic open"}
        </button>
        <p class="mic-hint">${$?"Mic muted for new questions — you can still barge in to stop Kaki mid-speech. Unmute to ask.":"Listening while ON. Speak over Kaki to interrupt. Headphones improve ASR + barge-in."}</p>

        <div class="badges">${s}</div>

        <div class="actions">
          <button type="button" id="probe" ${M?"":"disabled"}>Probe TTS</button>
          <button type="button" id="say" ${M?"":"disabled"}>Send text turn</button>
          <button type="button" id="refresh" ${S?"":"disabled"}>Refresh</button>
        </div>

        <div class="side-panel">
          <h2>Timers</h2>
          ${n}
        </div>
        <div class="side-panel">
          <h2>Notes</h2>
          ${t}
        </div>
        ${De()}
      </aside>

      <main class="col col-conversation">
        <section class="transcript-panel">
          <div class="panel-heading">
            <h2>Conversation</h2>
            <button
              type="button"
              class="copy-log ${Date.now()<K?"copied":""}"
              id="copy-log"
              ${i.length===0?"disabled":""}
              title="Copy conversation log (oldest → newest)"
            >${Date.now()<K?"Copied":"Copy log"}</button>
          </div>
          <div class="transcript" id="transcript">${b}</div>
        </section>
        <details class="diagnostics">
          <summary>Diagnostics</summary>
          <div class="token-bar">
            <input id="token" type="password" placeholder="Override UI token (optional)" value="" />
            <button type="button" id="save-token">Save override</button>
          </div>
          <pre>${d(JSON.stringify(l,null,2))}</pre>
        </details>
      </main>
    </div>
  `,He(),Se(e),Ne()}function He(){var e,s,n,t,i,o,a;(e=document.querySelector("#save-token"))==null||e.addEventListener("click",()=>{const r=document.querySelector("#token").value.trim();r&&(G=r,W(r)),te()}),(s=document.querySelector("#toggle"))==null||s.addEventListener("click",()=>{H({type:"set_enabled",enabled:!l.enabled})}),(n=document.querySelector("#mic-mute"))==null||n.addEventListener("click",()=>{H({type:"set_mic_muted",muted:!l.mic_muted})}),(t=document.querySelector("#probe"))==null||t.addEventListener("click",()=>{H({type:"probe",text:"Session is on."})}),(i=document.querySelector("#say"))==null||i.addEventListener("click",async()=>{const r=window.prompt("User text turn");if(r){L=!0,c();try{await fetch("/api/say",{method:"POST",headers:X(!0),body:JSON.stringify({text:r})})}catch{L=!1,c()}}}),(o=document.querySelector("#refresh"))==null||o.addEventListener("click",()=>{H({type:"get_session"}),y()}),(a=document.querySelector("#copy-log"))==null||a.addEventListener("click",()=>{Ce().then(r=>{if(!r){v("Could not copy conversation log","warn");return}K=Date.now()+1600,D!=null&&window.clearTimeout(D),D=window.setTimeout(()=>{K=0,D=null,c()},1600),v("Conversation log copied","ok")})}),document.querySelectorAll("[data-job-reply]").forEach(r=>{r.addEventListener("input",()=>{const u=r.dataset.jobReply;u&&(A[u]=r.value)})}),document.querySelectorAll("[data-job-reply-send]").forEach(r=>{r.addEventListener("click",()=>{const u=r.dataset.jobReplySend;u&&le(u,A[u]||"",!1)})}),document.querySelectorAll("[data-job-approve]").forEach(r=>{r.addEventListener("click",()=>{const u=r.dataset.jobApprove;if(!u)return;const b=(A[u]||"").trim()||"approved";le(u,b,!0)})}),document.querySelectorAll("[data-job-cancel]").forEach(r=>{r.addEventListener("click",()=>{const u=r.dataset.jobCancel;u&&(R=u,c())})}),document.querySelectorAll("[data-job-cancel-no]").forEach(r=>{r.addEventListener("click",()=>{R=null,c()})}),document.querySelectorAll("[data-job-cancel-yes]").forEach(r=>{r.addEventListener("click",()=>{const u=r.dataset.jobCancelYes;u&&Ke(u)})})}async function le(e,s,n){if(_==null){_=e,g="",c();try{const t=await fetch(`/api/jobs/${e}/respond`,{method:"POST",headers:X(!0),body:JSON.stringify({response:s,approve:n})});if(!t.ok){const a=await fe(t);g=a,v(a,"warn",e);return}const i=await t.json(),o=i.job||i;o!=null&&o.id&&Z(o),delete A[e],v(n?"Job approved":"Reply sent","ok",e),y()}catch{g="Could not reach the server",v(g,"warn",e)}finally{_=null,c()}}}async function Ke(e){if(_==null){_=e,g="",c();try{const s=await fetch(`/api/jobs/${e}/cancel`,{method:"POST",headers:X()});if(!s.ok){const i=await fe(s);g=i,v(i,"warn",e);return}const n=await s.json(),t=n.job||n;t!=null&&t.id&&Z(t),R=null,v("Job cancelled","info",e),y()}catch{g="Could not reach the server",v(g,"warn",e)}finally{_=null,R=null,c()}}}function d(e){return e.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}function p(e){return d(e).replace(/'/g,"&#39;")}function Ue(e){const s=e.state||"off";(s==="thinking"||s==="speaking"||s==="off")&&(L=!1),l={state:s,enabled:!!e.enabled,readiness:e.readiness||l.readiness,degraded:!!e.degraded,transcript:J(e.transcript||l.transcript),last_error:e.last_error,utterance_generation:e.utterance_generation,mic_muted:!!e.mic_muted},c()}function Be(e){var s;if(e.type==="session"||e.state)Ue(e),y();else if(e.type==="transcript")(e.role||"")==="assistant"&&(e.text||"").trim()&&(L=!1),qe(e),c();else if(e.type==="turn_done"||e.type==="turn_cancelled"||e.type==="error")L=!1,e.type==="error"&&e.message&&(l.last_error=String(e.message)),c();else if(e.type==="readiness")l.readiness=e.readiness||l.readiness,c();else if(e.type==="notification"){const n=String(((s=e.payload)==null?void 0:s.message)||e.message||"").trim();if(n){const t=String(e.status||"");v(n,t==="failed"?"warn":t==="completed"?"ok":"info",e.job_id?String(e.job_id):void 0)}y()}else if(e.type==="notification_delivered"||e.type==="tool_result")y();else if(e.type==="job_update"){const n=e.job||{};n.id&&(Z(n),c());const t=String(n.status||"");if((t==="completed"||t==="failed")&&!T.some(i=>i.jobId&&n.id&&i.jobId===String(n.id))){const i=String(n.specification||n.id||"job").split(`
`)[0].slice(0,72);v(t==="completed"?`Job finished: ${i}`:`Job failed: ${i}`,t==="failed"?"warn":"ok",n.id?String(n.id):void 0)}else n.id&&(t==="completed"||t==="failed")&&me(String(n.id));y()}}function H(e){h&&h.readyState===WebSocket.OPEN&&h.send(JSON.stringify(e))}function ce(){S=!1,l={...l,enabled:!1,state:"off"},c()}function Fe(){if(C)return;if(C=!0,h){try{h.onclose=null,h.close()}catch{}h=null}const e=Q(),n=`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/session?token=${encodeURIComponent(e)}`;h=new WebSocket(n),h.onopen=()=>{C=!1,S=!0,c(),y()},h.onclose=()=>{C=!1,ce(),setTimeout(()=>{te()},1500)},h.onerror=()=>{C=!1,ce()},h.onmessage=t=>{try{Be(JSON.parse(t.data))}catch{}}}async function te(){G=await ve(),c(),Fe()}function We(){O==null&&(O=window.setInterval(()=>{N.length||U()?c():O!=null&&(window.clearInterval(O),O=null)},1e3))}c();te();
