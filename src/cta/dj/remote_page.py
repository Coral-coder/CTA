"""The phone remote: one self-contained page (no external files, works offline
on the LAN). Served by dj.stream.StreamServer at /."""

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#121316"><meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"><meta name="apple-mobile-web-app-title" content="CTA">
<link rel="apple-touch-icon" href="/apple-touch-icon.png"><link rel="icon" type="image/png" href="/icon-512.png">
<link rel="manifest" href="/manifest.webmanifest">
<title>CTA live remote</title>
<style>
:root{--bg:#121316;--card:#1d1f24;--card2:#25272e;--fg:#eceef2;--mut:#9a9ca6;--acc:#3d8bff;--a:#2fa8ff;--b:#ff8a2f;
--ok:#3ddc84;--bad:#ff4d5e;--r:14px}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.35 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
body{padding:calc(12px + env(safe-area-inset-top)) 12px calc(20px + env(safe-area-inset-bottom));max-width:720px;margin:auto}
h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:18px 4px 8px;font-weight:700}
.card{background:var(--card);border-radius:var(--r);padding:12px}
button{font:inherit;color:var(--fg);background:var(--card2);border:0;border-radius:12px;padding:12px 14px;cursor:pointer}
button:active{transform:scale(.97)}button.pri{background:var(--acc);color:#fff;font-weight:700}
button.on{background:var(--ok);color:#06210f;font-weight:700}
input{font:inherit;color:var(--fg);background:var(--card2);border:1px solid #333640;border-radius:12px;padding:12px;width:100%}
.row{display:flex;gap:8px;align-items:center}.grow{flex:1;min-width:0}
.mut{color:var(--mut)}.small{font-size:12px}.ell{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#top{display:flex;align-items:center;gap:10px;margin-bottom:10px}
#top .logo{font-weight:900;font-size:18px;letter-spacing:.04em}
#np{display:flex;gap:14px;align-items:center}
#np img,#np .noart{width:112px;height:112px;border-radius:12px;object-fit:cover;background:#2a2c33;flex:none}
#np .t{font-size:19px;font-weight:800}#np .a{color:var(--mut);margin-top:2px}
.badge{display:inline-block;padding:2px 8px;border-radius:8px;background:#2a2c33;font-size:12px;font-weight:700;margin:6px 4px 0 0}
.bar{height:6px;background:#2a2c33;border-radius:3px;overflow:hidden;margin-top:10px}.bar i{display:block;height:100%;background:var(--acc)}
#status{margin-top:8px;font-size:13px;color:var(--mut)}
#transport{display:grid;grid-template-columns:1fr 1.3fr 1fr 1fr;gap:8px;margin-top:12px}
#transport button{font-size:22px;padding:14px 0}
#transport small{display:block;font-size:11px;color:var(--mut);margin-top:2px}#transport .pri small{color:#e3ecff}
#gear{width:100%;height:auto;display:block;border-radius:var(--r);background:#17181c;touch-action:pan-y}
.wv{position:relative;margin-top:8px}.wv canvas{width:100%;height:84px;display:block;border-radius:10px;background:#16171b;touch-action:none}
.wv .lab{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);margin:6px 2px 0}
.chips{display:flex;gap:6px;flex-wrap:wrap}.chips button{padding:8px 12px;font-size:14px}
.list .it{display:flex;gap:10px;align-items:center;padding:8px 4px;border-bottom:1px solid #26282f}
.list .it:last-child{border-bottom:0}
.list img,.list .noart{width:44px;height:44px;border-radius:8px;object-fit:cover;background:#2a2c33;flex:none}
.list .t{font-weight:700}.list .s{font-size:12px;color:var(--mut)}
.list .acts{display:flex;gap:4px}.list .acts button{padding:8px 10px;font-size:14px}
.played{opacity:.55}.pl{color:var(--ok);font-weight:700}
#gate{position:fixed;inset:0;background:var(--bg);display:none;align-items:center;justify-content:center;padding:24px;z-index:9}
#gate .card{max-width:360px;width:100%;text-align:center}
#toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);background:#2c2f38;padding:10px 16px;border-radius:12px;
display:none;z-index:10;max-width:90%;box-shadow:0 6px 24px #0008}
.hint{font-size:12px;color:var(--mut);margin-top:6px}
</style></head><body>
<div id="gate"><div class="card"><div style="font-size:42px">🎧</div><h3>Connect to CTA</h3>
<p class="mut">Enter the PIN shown in CTA on your computer (DJ decks → 📱 Phone remote).</p>
<input id="pin" inputmode="numeric" autocomplete="one-time-code" placeholder="PIN" maxlength="12">
<div style="height:10px"></div><button class="pri" style="width:100%" onclick="setPin()">Connect</button>
<p id="gerr" class="small" style="color:var(--bad)"></p></div></div>
<div id="toast"></div>

<div id="top"><span class="logo">CTA</span><span class="mut small grow ell" id="conn">connecting…</span>
<button id="listen" onclick="listen()">🔈 Listen</button></div>

<div class="card">
 <div id="np"><div class="noart" id="npart"></div><div class="grow">
  <div class="t ell" id="npt">Nothing playing</div><div class="a ell" id="npa">Pick a seed song below to start a set</div>
  <div id="npb"></div></div></div>
 <div class="bar"><i id="npbar" style="width:0"></i></div>
 <div class="row small mut" style="justify-content:space-between;margin-top:4px"><span id="npel"></span><span id="npre"></span></div>
 <div id="status">—</div>
 <div id="transport">
  <button onclick="cmd('back')">⏮<small>back</small></button>
  <button class="pri" id="pp" onclick="cmd('toggle_pause')">⏯<small>pause</small></button>
  <button onclick="cmd('skip',{mode:'mix'})">⏭<small>skip · mix</small></button>
  <button onclick="cmd('skip',{mode:'cut'})">⚡<small>cut now</small></button>
 </div>
 <div class="row" style="margin-top:8px"><button id="am" class="grow" onclick="cmd('automix',{on:!S.automix.active})">Auto-mix</button></div>
</div>

<h2>Decks</h2>
<canvas id="gear" width="720" height="330"></canvas>

<h2>Transition</h2>
<div class="card">
 <div class="row small"><span class="grow mut">Blend length (bars)</span><div class="chips" id="bars"></div></div>
 <div id="waves"></div>
 <div class="hint">Drag <b style="color:var(--ok)">IN</b> (where the next track starts) and <b style="color:var(--bad)">OUT</b>
 (where the playing track starts to leave). They snap to the start of the bar and are saved with the track.</div>
</div>

<h2>Up next</h2><div class="card list" id="queue"></div>

<h2>Start a set from a seed song</h2>
<div class="card"><input id="q" placeholder="Search artist, title, genre…" oninput="search()" autocomplete="off">
<div class="list" id="results"></div></div>

<h2>Played tonight</h2><div class="card list" id="hist"></div>
<audio id="aud" preload="none"></audio>

<script>
"use strict";
// the PIN is kept in the address (?pin=) so "Add to Home Screen" saves a link that connects by itself;
// home-screen apps on iPhone do not share storage with Safari
const STORED = localStorage.getItem('cta_pin') || '';
const hp = new URLSearchParams(location.hash.slice(1)), qp = new URLSearchParams(location.search);
let PIN = hp.get('pin') || qp.get('pin') || STORED, triedStored = false;
function keepPin() { localStorage.setItem('cta_pin', PIN); history.replaceState(null, '', PIN ? '/?pin=' + encodeURIComponent(PIN) : '/'); }
keepPin();
let S = null, lastT = 0, waves = {}, drag = null, gearAng = {A: 0, B: 0, C: 0, D: 0}, artImg = {};
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const fmt = t => { t = Math.max(0, t | 0); return (t / 60 | 0) + ':' + String(t % 60).padStart(2, '0'); };
const q = (u) => u + (u.includes('?') ? '&' : '?') + 'k=' + encodeURIComponent(PIN);
function toast(m) { const t = $('toast'); t.textContent = m; t.style.display = 'block'; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = 'none', 2600); }
function gate(show, err) { $('gate').style.display = show ? 'flex' : 'none'; $('gerr').textContent = err || ''; }
function setPin() { PIN = $('pin').value.trim(); keepPin(); gate(false); poll(); }

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {headers: {'X-CTA-Key': PIN}, cache: 'no-store'} :
    {method: 'POST', headers: {'X-CTA-Key': PIN, 'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  if (r.status === 401) {
    if (!triedStored && STORED && STORED !== PIN) { triedStored = true; PIN = STORED; keepPin(); return api(path, body); }
    gate(true, PIN ? 'Wrong PIN - CTA shows the current one under 📱 Phone remote' : ''); throw new Error('auth');
  }
  if (r.status === 429) { gate(true, 'Too many attempts - wait a minute'); throw new Error('auth'); }
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
async function cmd(action, args) {
  try { const r = await api('/api/cmd', Object.assign({action}, args || {})); if (r.message) toast(r.message); poll(); }
  catch (e) { if (e.message !== 'auth') toast('⚠ ' + e.message); }
}

const NOART = new Set();   // covers that do not exist: never ask again
function noArt(el) { NOART.add(el.dataset.u); const d = document.createElement('div'); d.className = 'noart'; el.replaceWith(d); }
function artTag(it) {
  return it && it.art && !NOART.has(it.art) ? `<img loading="lazy" data-u="${it.art}" src="${q(it.art)}" alt="" onerror="noArt(this)">`
    : `<div class="noart"></div>`;
}
function setHTML(el, html) { if (el._h !== html) { el._h = html; el.innerHTML = html; } }   // no flicker, no reloads
function trackRow(it, acts) {
  return `<div class="it ${it.played ? 'played' : ''}">${artTag(it)}<div class="grow"><div class="t ell">${esc(it.title)}</div>
  <div class="s ell">${[esc(it.artist), it.bpm ? it.bpm.toFixed(1) : '', esc(it.key)].filter(Boolean).join(' · ')}${it.played ? ' · <span class="pl">✓ played</span>' : ''}</div></div>
  <div class="acts">${acts}</div></div>`;
}

function render() {
  if (!S) return;
  $('conn').textContent = (S.listeners ? S.listeners + ' listening · ' : '') + 'connected';
  const m = S.decks.find(d => d.master) || S.decks.find(d => d.playing);
  if (m) {
    $('npt').textContent = m.title || '—'; $('npa').textContent = m.artist || '';
    $('npb').innerHTML = `<span class="badge">${m.bpm.toFixed(1)} BPM</span><span class="badge">${esc(m.key)}</span><span class="badge">Deck ${m.deck}</span>`;
    const ae = $('npart'); const src = m.art && !NOART.has(m.art) ? q(m.art) : '';
    if (src) { if (ae.tagName !== 'IMG' || ae.dataset.src !== src) { const i = new Image(); i.id = 'npart'; i.dataset.src = src; i.dataset.u = m.art; i.onerror = () => { NOART.add(m.art); render(); }; i.src = src; ae.replaceWith(i); } }
    else if (ae.tagName === 'IMG') { const d = document.createElement('div'); d.className = 'noart'; d.id = 'npart'; ae.replaceWith(d); }
  } else { $('npt').textContent = 'Nothing playing'; $('npa').textContent = 'Pick a seed song below to start a set'; $('npb').innerHTML = ''; }
  $('status').textContent = S.automix.status + (S.paused ? ' · PAUSED' : '');
  $('pp').innerHTML = (S.paused ? '▶' : '❚❚') + `<small>${S.paused ? 'resume' : 'pause'}</small>`;
  $('am').textContent = S.automix.active ? '■ Auto-mix on' : '▶ Start auto-mix'; $('am').className = 'grow ' + (S.automix.active ? 'on' : '');
  setHTML($('bars'), [4, 8, 16, 32].map(b => `<button class="${S.automix.blend_bars === b ? 'on' : ''}" onclick="cmd('blend',{bars:${b}})">${b}</button>`).join(''));
  setHTML($('queue'), S.queue.length ? S.queue.map((it, i) => trackRow(it,
    `<button title="play next" onclick="cmd('queue_play',{pos:${i}})">▶</button>` +
    (i ? `<button title="move up" onclick="cmd('queue_move',{pos:${i},to:${i - 1}})">↑</button>` : '') +
    `<button title="remove" onclick="cmd('queue_remove',{pos:${i}})">✕</button>`)).join('')
    : '<div class="mut small">The queue is empty - auto-mix picks the best next track itself, or choose a seed below.</div>');
  setHTML($('hist'), S.history.length ? S.history.slice().reverse().map(it => trackRow(it,
    `<button title="play next" onclick="cmd('play_next',{idx:${it.idx}})">▶</button>`)).join('') : '<div class="mut small">Nothing yet.</div>');
  renderWaves();
}

// ---- DJ gear --------------------------------------------------------------------------------------
function img(url) {
  if (!url || NOART.has(url)) return null;
  if (!artImg[url]) { const i = new Image(); i.onerror = () => NOART.add(url); i.src = q(url); artImg[url] = i; }
  return artImg[url];
}
function drawDeck(c, d, cx, cy, R, col) {
  c.save();
  c.fillStyle = '#23252b'; c.strokeStyle = '#0c0d10'; c.lineWidth = 2;
  c.beginPath(); c.roundRect(cx - R - 18, cy - R - 18, 2 * R + 36, 2 * R + 36, 14); c.fill(); c.stroke();
  // platter + record
  c.fillStyle = '#0b0b0d'; c.beginPath(); c.arc(cx, cy, R, 0, 7); c.fill();
  c.strokeStyle = '#1c1d22'; c.lineWidth = 1;
  for (let k = 0.45; k < 0.98; k += 0.06) { c.beginPath(); c.arc(cx, cy, R * k, 0, 7); c.stroke(); }
  const a = gearAng[d ? d.deck : 'A'] || 0;
  c.translate(cx, cy); c.rotate(a);
  const lr = R * 0.4, im = d && d.loaded ? img(d.art) : null;
  c.save(); c.beginPath(); c.arc(0, 0, lr, 0, 7); c.clip();
  if (im && im.complete && im.naturalWidth) c.drawImage(im, -lr, -lr, 2 * lr, 2 * lr);
  else { c.fillStyle = d && d.loaded ? col : '#2a2c33'; c.fillRect(-lr, -lr, 2 * lr, 2 * lr); }
  c.restore();
  c.fillStyle = '#ffffffcc'; c.beginPath(); c.arc(0, -R * 0.82, 4, 0, 7); c.fill();     // marker dot shows the spin
  c.fillStyle = '#ddd'; c.beginPath(); c.arc(0, 0, 4, 0, 7); c.fill();
  c.setTransform(1, 0, 0, 1, 0, 0);
  // tone arm
  const on = d && d.loaded;
  c.strokeStyle = '#b8bcc6'; c.lineWidth = 5; c.lineCap = 'round';
  const px = cx + R + 6, py = cy - R + 4, ang = on ? 2.2 : 1.75, len = R * 1.05;
  c.beginPath(); c.moveTo(px, py); c.lineTo(px + Math.cos(ang) * len, py + Math.sin(ang) * len); c.stroke();
  c.fillStyle = '#555a66'; c.beginPath(); c.arc(px, py, 9, 0, 7); c.fill();
  // deck letter + play light
  c.fillStyle = col; c.font = 'bold 22px system-ui'; c.textAlign = 'left'; c.fillText(d ? d.deck : '', cx - R - 10, cy - R + 8);
  c.fillStyle = d && d.playing ? '#3ddc84' : '#3a3d46'; c.beginPath(); c.arc(cx - R - 2, cy + R - 4, 7, 0, 7); c.fill();
  c.restore();
}
function drawGear() {
  const cv = $('gear'), c = cv.getContext('2d'), W = cv.width, H = cv.height;
  c.clearRect(0, 0, W, H);
  const decks = S ? S.decks : [], A = decks.find(d => d.deck === 'A'), B = decks.find(d => d.deck === 'B');
  const R = 112, cy = 150;
  drawDeck(c, A || {deck: 'A'}, 150, cy, R, getComputedStyle(document.body).getPropertyValue('--a'));
  drawDeck(c, B || {deck: 'B'}, W - 150, cy, R, getComputedStyle(document.body).getPropertyValue('--b'));
  // mixer
  const mx = W / 2 - 62, mw = 124;
  c.fillStyle = '#23252b'; c.beginPath(); c.roundRect(mx, 20, mw, 270, 14); c.fill();
  [[A, mx + 34, '#2fa8ff'], [B, mx + mw - 34, '#ff8a2f']].forEach(([d, x, col]) => {
    c.fillStyle = '#0d0e11'; c.fillRect(x - 3, 60, 6, 150);
    const v = d && d.loaded ? d.volume : 0, y = 210 - v * 150;
    c.fillStyle = col; c.fillRect(x - 14, y - 7, 28, 14);
    const lv = d ? Math.min(1, d.level) : 0;                  // meter
    c.fillStyle = '#1b1c20'; c.fillRect(x + 16, 60, 5, 150);
    c.fillStyle = lv > .9 ? '#ff4d5e' : '#3ddc84'; c.fillRect(x + 16, 210 - lv * 150, 5, lv * 150);
  });
  c.fillStyle = '#0d0e11'; c.fillRect(mx + 16, 250, mw - 32, 6);
  const xf = S ? S.crossfader : 0, xx = mx + mw / 2 + xf * (mw / 2 - 26);
  c.fillStyle = '#e8e8ee'; c.fillRect(xx - 9, 241, 18, 24);
  c.fillStyle = '#9a9ca6'; c.font = '12px system-ui'; c.textAlign = 'center'; c.fillText('MIXER', mx + mw / 2, 42);
  // titles
  c.font = '14px system-ui'; c.fillStyle = '#eceef2';
  [[A, 150], [B, W - 150]].forEach(([d, x]) => {
    let t = d && d.loaded ? d.title : 'empty'; if (t.length > 28) t = t.slice(0, 27) + '…';
    c.fillText(t, x, H - 14);
  });
}
function spin(dt) {
  if (!S) return;
  for (const d of S.decks) if (d.playing) gearAng[d.deck] = ((gearAng[d.deck] || 0) + dt * (33.333 / 60) * 2 * Math.PI * (d.rate || 1)) % (2 * Math.PI);
}

// ---- transition waveforms ------------------------------------------------------------------------
async function getWave(d) {
  const key = d.deck + ':' + d.wid;
  if (waves[key] === undefined) { waves[key] = null; try { waves[key] = await api('/api/wave?deck=' + d.deck); renderWaves(); } catch (e) { delete waves[key]; } }
  return waves[key];
}
function renderWaves() {
  if (!S) return;
  const box = $('waves'), list = S.decks.filter(d => d.loaded && (d.playing || d.next)).sort((a, b) => b.master - a.master);
  const ids = list.map(d => d.deck).join(',');
  if (box.dataset.ids !== ids) {
    box.dataset.ids = ids;
    box.innerHTML = list.length ? list.map(d => `<div class="wv"><canvas id="wv${d.deck}" data-deck="${d.deck}"></canvas>
      <div class="lab"><span class="ell" id="wl${d.deck}"></span><span id="wr${d.deck}"></span></div></div>`).join('')
      : '<div class="mut small" style="margin-top:8px">Load or start a track to edit its transition markers.</div>';
    list.forEach(d => bindWave($('wv' + d.deck)));
  }
  list.forEach(d => {
    $('wl' + d.deck).innerHTML = `<b>${d.deck}</b> · ${esc(d.title)} ${d.master ? '<span class="pl">playing</span>' : '<span class="mut">next</span>'}`;
    getWave(d); drawWave(d);
  });
}
function deckOf(n) { return S && S.decks.find(d => d.deck === n); }
function drawWave(d) {
  const cv = $('wv' + d.deck); if (!cv) return;
  const w = waves[d.deck + ':' + d.wid];
  const W = cv.clientWidth * devicePixelRatio, H = cv.clientHeight * devicePixelRatio;
  if (cv.width !== W) { cv.width = W; cv.height = H; }
  const c = cv.getContext('2d'); c.clearRect(0, 0, W, H);
  if (!w) return;
  const dur = w.duration, X = t => t / dur * W, n = w.low.length, bw = W / n;
  for (let i = 0; i < n; i++) {
    const lo = w.low[i] / 99, mi = w.mid[i] / 99, hi = w.high[i] / 99, h = Math.max(lo, mi, hi) * H * 0.9;
    c.fillStyle = `rgb(${40 + 200 * lo | 0},${70 + 140 * mi | 0},${90 + 165 * hi | 0})`;
    c.fillRect(i * bw, (H - h) / 2, Math.max(bw, 1), h);
  }
  // bar lines every 8 bars
  if (w.bpm > 0) {
    const bar = 4 * 60 / w.bpm; c.fillStyle = '#ffffff22';
    for (let t = w.first_beat; t < dur; t += bar * 8) c.fillRect(X(t), 0, 1, H);
  }
  const mk = drag && drag.deck === d.deck ? drag : {in: d.mix_in, out: d.mix_out};
  // planned blend
  const tr = S.transition;
  if (tr && w.bpm > 0) {
    if (tr.out_deck === d.deck && tr.start_t != null) { c.fillStyle = '#ff4d5e33'; c.fillRect(X(tr.start_t), 0, X(tr.beats * 60 / w.bpm), H); }
    if (tr.in_deck === d.deck && tr.in_start_t != null) { c.fillStyle = '#3ddc8433'; c.fillRect(X(tr.in_start_t), 0, X(tr.beats * 60 / w.bpm), H); }
  }
  const flag = (t, col, txt) => { if (t == null) return; const x = X(t); c.fillStyle = col; c.fillRect(x - 1.5 * devicePixelRatio, 0, 3 * devicePixelRatio, H);
    c.font = `bold ${11 * devicePixelRatio}px system-ui`; const tw = c.measureText(txt).width + 10 * devicePixelRatio;
    c.fillRect(Math.min(x, W - tw), 0, tw, 16 * devicePixelRatio); c.fillStyle = '#111'; c.fillText(txt, Math.min(x, W - tw) + 5 * devicePixelRatio, 12 * devicePixelRatio); };
  flag(mk.in, '#3ddc84', 'IN'); flag(mk.out, '#ff4d5e', 'OUT');
  c.fillStyle = '#fff'; c.fillRect(X(d.time) - 1, 0, 2 * devicePixelRatio, H);
  $('wr' + d.deck).textContent = fmt(d.time) + ' / ' + fmt(dur);
}
function snapBar(w, t) {
  if (!(w.bpm > 0)) return t;
  const bar = 4 * 60 / w.bpm; return Math.max(0, w.first_beat + Math.floor((t - w.first_beat) / bar + 1e-6) * bar);
}
function bindWave(cv) {
  const deck = cv.dataset.deck;
  const tAt = ev => { const r = cv.getBoundingClientRect(), d = deckOf(deck), w = d && waves[deck + ':' + d.wid];
    return w ? Math.min(Math.max((ev.clientX - r.left) / r.width, 0), 1) * w.duration : 0; };
  cv.addEventListener('pointerdown', ev => {
    const d = deckOf(deck), w = d && waves[deck + ':' + d.wid]; if (!w) return;
    const t = tAt(ev), tol = w.duration * 0.04;
    const di = d.mix_in != null ? Math.abs(t - d.mix_in) : 1e9, dO = d.mix_out != null ? Math.abs(t - d.mix_out) : 1e9;
    let which = di <= dO ? (di < tol ? 'in' : null) : (dO < tol ? 'out' : null);
    if (!which) which = d.master ? 'out' : 'in';             // tap empty space: the playing track's OUT, the next track's IN
    drag = {deck, which, in: d.mix_in, out: d.mix_out}; drag[which] = snapBar(w, t);
    cv.setPointerCapture(ev.pointerId); drawWave(d); ev.preventDefault();
  });
  cv.addEventListener('pointermove', ev => {
    if (!drag || drag.deck !== deck) return; const d = deckOf(deck), w = waves[deck + ':' + d.wid];
    drag[drag.which] = snapBar(w, tAt(ev)); drawWave(d);
  });
  const end = () => { if (!drag || drag.deck !== deck) return; const g = drag; drag = null;
    if (g.in != null && g.out != null && g.out <= g.in) { toast('OUT must come after IN'); renderWaves(); return; }
    cmd('markers', {deck, in: g.in, out: g.out}); };
  cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', () => { drag = null; renderWaves(); });
}

// ---- search / seed -----------------------------------------------------------------------------
let sTimer = null;
function search() {
  clearTimeout(sTimer);
  sTimer = setTimeout(async () => {
    const v = $('q').value.trim(); if (!v) { $('results').innerHTML = ''; return; }
    try {
      const r = await api('/api/search?q=' + encodeURIComponent(v));
      $('results').innerHTML = r.results.length ? r.results.map(it => trackRow(it,
        `<button class="pri" title="start a new set from this song" onclick="cmd('seed',{idx:${it.idx}})">Seed</button>` +
        `<button title="play next" onclick="cmd('play_next',{idx:${it.idx}})">▶</button>` +
        `<button title="add to queue" onclick="cmd('queue_add',{idx:${it.idx}})">＋</button>`)).join('') : '<div class="mut small">No matches.</div>';
    } catch (e) {}
  }, 250);
}

// ---- listen ----------------------------------------------------------------------------------------
function listen() {
  const a = $('aud'), b = $('listen');
  if (!a.paused) { a.pause(); a.removeAttribute('src'); a.load(); b.textContent = '🔈 Listen'; b.className = ''; return; }
  a.src = q('/live.mp3?' + Date.now()); a.play().catch(() => toast('Tap again to start audio'));
  b.textContent = '🔊 Stop'; b.className = 'on';
}

// ---- loop ----------------------------------------------------------------------------------------------
async function poll() {
  try { S = await api('/api/state'); lastT = performance.now(); gate(false); render(); }
  catch (e) { if (e.message !== 'auth') $('conn').textContent = 'reconnecting…'; }
}
let prev = performance.now();
function frame(now) {
  const dt = (now - prev) / 1000; prev = now;
  if (S) {
    for (const d of S.decks) if (d.playing && !S.paused) d.time = Math.min(d.duration, d.time + dt * (d.rate || 1));
    spin(S.paused ? 0 : dt); drawGear();
    const m = S.decks.find(d => d.master) || S.decks.find(d => d.playing);
    if (m) { $('npbar').style.width = (100 * m.time / Math.max(m.duration, 1)) + '%'; $('npel').textContent = fmt(m.time); $('npre').textContent = '-' + fmt(m.duration - m.time); }
    if (!drag) for (const d of S.decks) if (d.loaded && (d.playing || d.next)) drawWave(d);
  }
  requestAnimationFrame(frame);
}
setInterval(() => { if (!document.hidden) poll(); }, 1000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
poll(); requestAnimationFrame(frame);
</script></body></html>"""
