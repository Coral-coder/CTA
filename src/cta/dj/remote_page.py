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
.wv .lab{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);margin:10px 2px 4px}
.wv canvas{height:140px}
.wvc{display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap}.wvc button{padding:9px 11px;font-size:14px}
.wv .lab{gap:8px;align-items:center}.zm{display:flex;gap:4px;flex:none}.zm button{padding:6px 11px;font-size:15px}
.seg{display:flex;gap:4px}.seg button{border:2px solid transparent;color:var(--c);font-weight:800}
.seg button.sel{border-color:var(--c);background:#2c2f38}
.chips button.sel{outline:2px solid var(--acc)}
button.mini{padding:6px 10px;font-size:12px;letter-spacing:0;text-transform:none}
h2.row{display:flex;align-items:center}
.list .it{touch-action:pan-y;transition:transform .15s}.list .it.swiping{transition:none}
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
#alert{display:none;background:#5a1620;color:#ffd9de;border-radius:12px;padding:10px 12px;margin-bottom:10px;font-weight:600}
</style></head><body>
<div id="gate"><div class="card"><div style="font-size:42px">🎧</div><h3>Connect to CTA</h3>
<p class="mut">Enter the PIN shown in CTA on your computer (DJ decks → 📱 Phone remote).</p>
<input id="pin" inputmode="numeric" autocomplete="one-time-code" placeholder="PIN" maxlength="12">
<div style="height:10px"></div><button class="pri" style="width:100%" onclick="setPin()">Connect</button>
<p id="gerr" class="small" style="color:var(--bad)"></p></div></div>
<div id="toast"></div>

<div id="alert"></div>
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
 <div class="small mut" id="blabel">Blend length (bars)</div><div class="chips" id="bars" style="margin-top:6px"></div>
 <div id="waves"></div>
 <div class="hint"><b style="color:var(--ok)">IN</b>: where the next track starts · <b style="color:var(--bad)">OUT</b>: where the
 playing track starts to leave · <b style="color:#ff9800">END</b>: where it is gone (sets the blend length).
 Pick one and tap a bar, drag a flag, or nudge it −4/+4 beats. Pinch or ＋/− to zoom, swipe to scroll.
 Markers snap to the start of the bar and are saved with the track.</div>
</div>

<h2 class="row">Up next<span class="grow"></span><button class="mini" onclick="clearQueue()">Clear queue</button></h2>
<div class="card list" id="queue"></div>

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
  return `<div class="it ${it.played ? 'played' : ''}" ${it.pos != null ? `data-pos="${it.pos}"` : ''}>${artTag(it)}<div class="grow"><div class="t ell">${esc(it.title)}</div>
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
  const au = S.automix.auto_length;
  $('blabel').textContent = au ? `Blend length: auto, up to ${S.automix.blend_bars} bars (never dead air)` : `Blend length: exactly ${S.automix.blend_bars} bars`;
  setHTML($('bars'), `<button class="${au ? 'on' : ''}" onclick="cmd('blend',{auto:${!au}})">Auto</button>` +
    [4, 8, 16, 32, 64, 128].map(b => `<button class="${S.automix.blend_bars === b ? 'sel' : ''}" onclick="cmd('blend',{bars:${b}})">${b}</button>`).join(''));
  setHTML($('queue'), S.queue.length ? S.queue.map((it, i) => trackRow(Object.assign({pos: i}, it),
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

// ---- transition waveforms: zoom (pinch / buttons), drag or bump IN / OUT / OUT END ----------------
const VIEW = {}, PICK = {};
const MK = {in: ['IN', '#3ddc84'], out: ['OUT', '#ff4d5e'], out_end: ['END', '#ff9800']};
async function getWave(d) {
  const key = d.deck + ':' + d.wid;
  if (waves[key] === undefined) { waves[key] = null; try { waves[key] = await api('/api/wave?bins=4000&deck=' + d.deck); renderWaves(); } catch (e) { delete waves[key]; } }
  return waves[key];
}
function deckOf(n) { return S && S.decks.find(d => d.deck === n); }
function waveOf(d) { return d && waves[d.deck + ':' + d.wid]; }
function viewOf(d, w) { const v = VIEW[d.deck]; return v && v.wid === d.wid ? v : {wid: d.wid, v0: 0, v1: w.duration}; }
function setView(d, w, v0, v1) {
  const barLen = w.bpm > 0 ? 240 / w.bpm : 2, span = Math.min(Math.max(v1 - v0, barLen * 4), w.duration);
  v0 = Math.min(Math.max(v0, 0), Math.max(w.duration - span, 0));
  VIEW[d.deck] = {wid: d.wid, v0, v1: v0 + span};
}
function zoomWave(deck, f, around) {
  const d = deckOf(deck), w = waveOf(d); if (!w) return;
  const v = viewOf(d, w), span = v.v1 - v.v0;
  let c = around;
  if (c == null) { const m = d[pickOf(d) === 'out_end' ? 'mix_out_end' : pickOf(d) === 'out' ? 'mix_out' : 'mix_in'];
    c = m != null && m >= v.v0 && m <= v.v1 ? m : (v.v0 + v.v1) / 2; }
  const ns = span * f, k = (c - v.v0) / span;
  setView(d, w, c - k * ns, c - k * ns + ns); drawWave(d);
}
function fitWave(deck) { delete VIEW[deck]; const d = deckOf(deck); if (d) drawWave(d); }
function pickOf(d) { return PICK[d.deck] || (d.master ? 'out' : 'in'); }
function setPick(deck, k) { PICK[deck] = k; renderWaves(true); }
function bump(deck, beats) { const d = deckOf(deck); if (d) cmd('bump', {deck, which: pickOf(d), beats}); }
function outLen(deck, bars) { cmd('out_length', {deck, bars}); }
function renderWaves(force) {
  if (!S) return;
  const box = $('waves'), list = S.decks.filter(d => d.loaded && (d.playing || d.next)).sort((a, b) => b.master - a.master);
  const ids = list.map(d => d.deck + d.wid).join(',');
  if (box.dataset.ids !== ids) {
    box.dataset.ids = ids;
    box.innerHTML = list.length ? list.map(d => `<div class="wv">
      <div class="lab"><span class="ell grow" id="wl${d.deck}"></span><span class="zm">
        <button onclick="zoomWave('${d.deck}',1.6)" title="zoom out">−</button><button onclick="zoomWave('${d.deck}',0.55)" title="zoom in">＋</button><button onclick="fitWave('${d.deck}')" title="whole track">⤢</button></span></div>
      <canvas id="wv${d.deck}" data-deck="${d.deck}"></canvas>
      <div class="wvc"><div class="seg" id="seg${d.deck}"></div>
        <button onclick="bump('${d.deck}',-4)" title="4 beats earlier">−4</button><button onclick="bump('${d.deck}',4)" title="4 beats later">+4</button>
        <span class="grow"></span><span class="small mut" id="wr${d.deck}"></span></div>
      <div class="wvc small" id="ol${d.deck}"></div></div>`).join('')
      : '<div class="mut small" style="margin-top:8px">Load or start a track to edit its transition markers.</div>';
    list.forEach(d => bindWave($('wv' + d.deck)));
  }
  list.forEach(d => {
    setHTML($('wl' + d.deck), `<b>${d.deck}</b> · ${esc(d.title)} ${d.master ? '<span class="pl">playing</span>' : '<span class="mut">next</span>'}`);
    const p = pickOf(d);
    setHTML($('seg' + d.deck), Object.keys(MK).map(k => `<button class="${p === k ? 'sel' : ''}" style="--c:${MK[k][1]}" onclick="setPick('${d.deck}','${k}')">${MK[k][0]}</button>`).join(''));
    const ob = d.mix_out != null && d.mix_out_end != null && waveOf(d) && waveOf(d).bpm > 0 ? Math.round((d.mix_out_end - d.mix_out) * waveOf(d).bpm / 240) : null;
    setHTML($('ol' + d.deck), d.mix_out == null ? '<span class="mut">Set OUT to choose an outro length</span>' : '<span class="mut">Outro</span>' +
      [['Auto', 0], ['16', 16], ['32', 32], ['64', 64], ['128', 128]].map(([t, b]) => `<button class="${(b ? ob === b : ob == null) ? 'on' : ''}" onclick="outLen('${d.deck}',${b || 'null'})">${t}</button>`).join(''));
    getWave(d); drawWave(d);
  });
}
function drawWave(d) {
  const cv = $('wv' + d.deck); if (!cv) return;
  const w = waveOf(d), dpr = devicePixelRatio;
  const W = cv.clientWidth * dpr, H = cv.clientHeight * dpr;
  if (cv.width !== W) { cv.width = W; cv.height = H; }
  const c = cv.getContext('2d'); c.clearRect(0, 0, W, H);
  if (!w) return;
  const v = viewOf(d, w), span = v.v1 - v.v0, X = t => (t - v.v0) / span * W, n = w.low.length, dur = w.duration;
  const top = 18 * dpr, wh = H - top - 14 * dpr, step = Math.max(1, Math.round(dpr));
  for (let x = 0; x < W; x += step) {
    const t0 = v.v0 + x / W * span, t1 = v.v0 + (x + step) / W * span;
    let i0 = Math.floor(t0 / dur * n), i1 = Math.max(i0 + 1, Math.ceil(t1 / dur * n));
    if (i0 >= n || i1 <= 0) continue;
    i0 = Math.max(i0, 0); i1 = Math.min(i1, n);
    let lo = 0, mi = 0, hi = 0;
    for (let i = i0; i < i1; i++) { lo = Math.max(lo, w.low[i]); mi = Math.max(mi, w.mid[i]); hi = Math.max(hi, w.high[i]); }
    lo /= 99; mi /= 99; hi /= 99;
    const h = Math.max(lo, mi, hi) * wh;
    c.fillStyle = `rgb(${40 + 200 * lo | 0},${70 + 140 * mi | 0},${90 + 165 * hi | 0})`;
    c.fillRect(x, top + (wh - h) / 2, step, h);
  }
  // bars: phrase lines and bar numbers, beats when close enough
  if (w.bpm > 0) {
    const bar = 240 / w.bpm, ppb = bar / span * W, every = [1, 4, 8, 16, 32, 64].find(k => ppb * k >= 34 * dpr) || 128;
    c.font = `${10 * dpr}px system-ui`;
    const b0 = Math.max(0, Math.floor((v.v0 - w.first_beat) / bar)), b1 = Math.ceil((v.v1 - w.first_beat) / bar);
    for (let b = b0; b <= b1; b++) {
      const x = X(w.first_beat + b * bar);
      if (b % every === 0) {
        c.fillStyle = b % 8 === 0 ? '#ffffff66' : '#ffffff30'; c.fillRect(x, top, Math.max(1, dpr * (b % 8 === 0 ? 1.5 : 1)), wh);
        c.fillStyle = '#9a9ca6'; c.fillText(String(b + 1), x + 2 * dpr, H - 3 * dpr);
      }
      if (ppb >= 90 * dpr) { c.fillStyle = '#ffffff18'; for (let k = 1; k < 4; k++) c.fillRect(X(w.first_beat + (b + k / 4) * bar), top, 1, wh); }
    }
  }
  const mk = drag && drag.deck === d.deck ? drag : {in: d.mix_in, out: d.mix_out, out_end: d.mix_out_end};
  const tr = S.transition, bl = w.bpm > 0 ? 60 / w.bpm : 0;
  if (mk.out != null && bl) {      // the outgoing blend: OUT .. END (or the blend length)
    const e = mk.out_end != null ? mk.out_end : mk.out + S.automix.blend_bars * 4 * bl;
    c.fillStyle = '#ff4d5e26'; c.fillRect(X(mk.out), top, X(e) - X(mk.out), wh);
  }
  if (tr && bl) {
    if (tr.in_deck === d.deck && tr.in_start_t != null) { c.fillStyle = '#3ddc8426'; c.fillRect(X(tr.in_start_t), top, tr.beats * bl / span * W, wh); }
  }
  const p = pickOf(d);
  for (const k of ['in', 'out', 'out_end']) {
    const t = mk[k]; if (t == null) continue;
    const x = X(t), [txt, col] = MK[k], sel = p === k;
    c.fillStyle = col; c.fillRect(x - (sel ? 2 : 1) * dpr, top, (sel ? 4 : 2) * dpr, wh);
    c.font = `bold ${11 * dpr}px system-ui`;
    const label = txt + (w.bpm > 0 ? ' ' + (Math.round((t - w.first_beat) * w.bpm / 240) + 1) : ''), tw = c.measureText(label).width + 10 * dpr;
    let lx = Math.min(Math.max(x - (k === 'out_end' ? tw : 0), 0), W - tw);
    if (x < 0 || x > W) {           // off screen: point at it instead of pretending it is here
      const arrow = x < 0 ? '◂ ' : ' ▸'; const lab2 = x < 0 ? arrow + label : label + arrow, tw2 = c.measureText(lab2).width + 10 * dpr;
      lx = x < 0 ? 0 : W - tw2; c.globalAlpha = 0.75; c.fillRect(lx, 0, tw2, top - 2 * dpr);
      c.fillStyle = '#111'; c.fillText(lab2, lx + 5 * dpr, top - 6 * dpr); c.globalAlpha = 1; continue;
    }
    c.fillRect(lx, 0, tw, top - 2 * dpr); if (sel) { c.strokeStyle = '#fff'; c.lineWidth = dpr; c.strokeRect(lx, 0, tw, top - 2 * dpr); }
    c.fillStyle = '#111'; c.fillText(label, lx + 5 * dpr, top - 6 * dpr);
  }
  c.fillStyle = '#fff'; c.fillRect(X(d.time) - dpr / 2, top, 2 * dpr, wh);
  const zoomed = span < dur * 0.98;
  setHTML($('wr' + d.deck), (zoomed ? `<span class="pl">${Math.max(1, Math.round(dur / span))}×</span> · ` : '') + fmt(d.time) + ' / ' + fmt(dur));
}
function snapBar(w, t) {
  if (!(w.bpm > 0)) return t;
  const bar = 240 / w.bpm; return Math.max(0, w.first_beat + Math.round((t - w.first_beat) / bar) * bar);
}
function bindWave(cv) {
  const deck = cv.dataset.deck, pts = new Map();
  let g = null;                                        // gesture
  const tAt = (x) => { const r = cv.getBoundingClientRect(), d = deckOf(deck), w = waveOf(d), v = viewOf(d, w);
    return v.v0 + Math.min(Math.max((x - r.left) / r.width, 0), 1) * (v.v1 - v.v0); };
  cv.addEventListener('pointerdown', ev => {
    const d = deckOf(deck), w = waveOf(d); if (!w) return;
    cv.setPointerCapture(ev.pointerId); pts.set(ev.pointerId, ev.clientX); ev.preventDefault();
    const v = viewOf(d, w);
    if (pts.size === 2) {                              // pinch to zoom
      const xs = [...pts.values()]; drag = null;
      g = {pinch: true, d0: Math.abs(xs[0] - xs[1]) || 1, c: tAt((xs[0] + xs[1]) / 2), v0: v.v0, v1: v.v1, rel: ((xs[0] + xs[1]) / 2 - cv.getBoundingClientRect().left) / cv.getBoundingClientRect().width};
      return;
    }
    const r = cv.getBoundingClientRect(), px = ev.clientX - r.left;
    let hit = null, best = 16;
    for (const [k, f] of [['in', 'mix_in'], ['out', 'mix_out'], ['out_end', 'mix_out_end']]) {
      if (d[f] == null) continue; const x = (d[f] - v.v0) / (v.v1 - v.v0) * r.width;
      if (Math.abs(x - px) < best) { best = Math.abs(x - px); hit = k; }
    }
    g = {x0: ev.clientX, v0: v.v0, v1: v.v1, hit, moved: false};
    if (hit) { PICK[deck] = hit; drag = {deck, which: hit, in: d.mix_in, out: d.mix_out, out_end: d.mix_out_end}; renderWaves(true); }
  });
  cv.addEventListener('pointermove', ev => {
    if (!pts.has(ev.pointerId) || !g) return;
    pts.set(ev.pointerId, ev.clientX);
    const d = deckOf(deck), w = waveOf(d); if (!w) return;
    if (g.pinch && pts.size >= 2) {
      const xs = [...pts.values()], dist = Math.abs(xs[0] - xs[1]) || 1, span = (g.v1 - g.v0) * g.d0 / dist;
      setView(d, w, g.c - g.rel * span, g.c - g.rel * span + span); drawWave(d); return;
    }
    if (Math.abs(ev.clientX - g.x0) > 6) g.moved = true;
    if (drag && drag.deck === deck) {
      const t = snapBar(w, tAt(ev.clientX));
      if (drag.which === 'out' && d.mix_out != null && d.mix_out_end != null) drag.out_end = d.mix_out_end + (t - d.mix_out);
      drag[drag.which] = t; drawWave(d);
    } else if (g.moved) {                              // scroll the zoomed view
      const r = cv.getBoundingClientRect(), dt = (g.x0 - ev.clientX) / r.width * (g.v1 - g.v0);
      setView(d, w, g.v0 + dt, g.v1 + dt); drawWave(d);
    }
  });
  const up = ev => {
    pts.delete(ev.pointerId);
    if (!g) return;
    const d = deckOf(deck), w = waveOf(d), gg = g;
    if (gg.pinch) { if (pts.size === 0) g = null; return; }
    g = null;
    if (drag && drag.deck === deck) {
      const m = drag; drag = null;
      const body = {deck}; body[m.which] = m[m.which]; if (m.which === 'out' && m.out_end != null) body.out_end = m.out_end;
      cmd('markers', body);
    } else if (!gg.moved && w) {                       // tap: put the selected marker on that bar
      const k = pickOf(d), body = {deck}; body[k] = snapBar(w, tAt(ev.clientX));
      if (k === 'out' && d.mix_out != null && d.mix_out_end != null) body.out_end = d.mix_out_end + (body.out - d.mix_out);
      cmd('markers', body);
    }
  };
  cv.addEventListener('pointerup', up);
  cv.addEventListener('pointercancel', ev => { pts.delete(ev.pointerId); g = null; drag = null; renderWaves(true); });
}

// ---- queue: swipe a row left to remove it, or clear everything -------------------------------------
function clearQueue() { if (S && S.queue.length && confirm('Remove every track from the queue?')) cmd('queue_clear'); }
(function swipeToRemove() {
  const box = $('queue'); let sw = null;
  box.addEventListener('pointerdown', ev => {
    const row = ev.target.closest('.it[data-pos]'); if (!row || ev.target.closest('button')) return;
    sw = {row, x0: ev.clientX, y0: ev.clientY, dx: 0, on: false};
  });
  box.addEventListener('pointermove', ev => {
    if (!sw) return; const dx = ev.clientX - sw.x0, dy = ev.clientY - sw.y0;
    if (!sw.on && Math.abs(dx) > 12 && Math.abs(dx) > Math.abs(dy) * 1.5) { sw.on = true; sw.row.classList.add('swiping'); }
    if (sw.on) { sw.dx = Math.min(0, dx); sw.row.style.transform = `translateX(${sw.dx}px)`; sw.row.style.opacity = 1 + sw.dx / 300; }
  });
  const end = () => {
    if (!sw) return; const s = sw; sw = null; s.row.classList.remove('swiping');
    if (s.on && s.dx < -90) { s.row.style.transform = 'translateX(-110%)'; cmd('queue_remove', {pos: +s.row.dataset.pos}); }
    else { s.row.style.transform = ''; s.row.style.opacity = ''; }
  };
  box.addEventListener('pointerup', end); box.addEventListener('pointercancel', end);
})();

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
let frames = null, framesT = 0;
function alertMsg(m) { const a = $('alert'); a.style.display = m ? 'block' : 'none'; a.textContent = m || ''; }
async function poll() {
  try {
    S = await api('/api/state'); lastT = performance.now(); gate(false);
    // the PC answers but its audio is not moving: say so instead of pretending
    if (S.frames !== frames) { frames = S.frames; framesT = lastT; }
    const playing = !S.paused && S.decks.some(d => d.playing);
    alertMsg(playing && lastT - framesT > 3500 ? '⚠ The mix has stopped on the PC (audio device not responding). CTA is trying to reconnect it…' : '');
    render();
  } catch (e) {
    if (e.message === 'auth') return;
    $('conn').textContent = 'reconnecting…';
    if (performance.now() - lastT > 4000) alertMsg('⚠ Lost connection to CTA - is the PC on, awake and on this Wi-Fi?');
  }
}
let prev = performance.now();
function frame(now) {
  let dt = (now - prev) / 1000; prev = now;
  const live = S && now - lastT < 4000 && $('alert').style.display !== 'block';   // never animate stale state
  if (S) {
    if (!live) dt = 0;
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
