// Dashboard: websocket client, 4-step wizard, drawing. All numbers come from the server.
(() => {
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const post = (url, body) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const cv = $('cv'), g = cv.getContext('2d');
  const SPAN = 8;                                   // seconds shown in the traces
  let forms = null, img = null, last = null, histPsi = [], histScore = [], setup = null, step = 1, status = {};
  let tool = null, pts = { tip: null, feed: null, work: null, s1: null, s2: null }, applied = false, idleOk = false;

  const chip = (id, txt, cls) => { const e = $(id); e.textContent = txt; e.className = 'chip ' + (cls || ''); };
  function go(n) {
    step = n; $('main').dataset.step = n;
    document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('on', +b.dataset.step === n));
    $('vid-title').textContent = n === 2 ? 'Camera: click to calibrate' : 'Camera';
    $('vid-help').textContent = n === 2 ? 'Place each point with the buttons on the right.' : 'Yellow = measured chip direction · blue = feed direction · ring = tool tip';
    if (n === 4) loadSummary();
    if (n === 1 && forms && setup) requestAnimationFrame(() => window.drawChipChart($('chart'), forms, setup.params));
    draw();
  }

  // ------------------------------------------------------------ websocket
  function connect() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/dash`);
    ws.binaryType = 'blob';
    ws.onmessage = async ev => {
      if (typeof ev.data === 'string') return onJson(JSON.parse(ev.data));
      const bmp = await createImageBitmap(ev.data.slice(1, ev.data.size, 'image/jpeg'));
      if (cv.width !== bmp.width) { cv.width = bmp.width; cv.height = bmp.height; }
      img = bmp; $('ph').hidden = true; draw();
    };
    ws.onclose = () => setTimeout(connect, 1000);
  }

  function onJson(j) {
    if (j.type === 'status') {
      status = j;
      chip('c-phone', 'phone: ' + j.phone, j.phone === 'connected' ? 'ok' : '');
      chip('c-mic', 'mic: ' + (j.mic ? 'on' : 'off'), j.mic ? 'ok' : 'warn');
      chip('c-cal', j.calibrated ? 'calibrated' : 'not calibrated', j.calibrated ? 'ok' : '');
      $('b-replay').hidden = j.mode !== 'replay';
      if (j.mode === 'replay' && step < 3) go(3);
    } else if (j.type === 'tilt') {
      const b = j.tilt.beta, gm = j.tilt.gamma, lim = 20;
      $('bubble').firstElementChild.style.left = (21 + Math.max(-lim, Math.min(lim, gm)) * 0.8) + 'px';
      $('bubble').firstElementChild.style.top = (21 + Math.max(-lim, Math.min(lim, b)) * 0.8) + 'px';
      const lvl = Math.abs(b) < 5 && Math.abs(gm) < 5;
      $('bubble').classList.toggle('level', lvl);
      $('tilt-txt').textContent = lvl ? 'level' : `tilt ${b.toFixed(0)}° / ${gm.toFixed(0)}°`;
      $('k-place').classList.toggle('done', lvl && status.phone === 'connected');
    } else if (j.type === 'state') { last = j; onState(j); draw(); }
  }

  function onState(j) {
    const c = j.chip, a = j.audio, t = j.t;
    chip('c-fps', j.fps.toFixed(0) + ' fps', j.fps >= 20 ? 'ok' : 'warn');
    chip('c-lat', 'latency ' + j.latency_ms.toFixed(0) + ' ms', j.latency_ms < 300 ? 'ok' : 'warn');
    $('psi').textContent = c.psi_display == null ? '--' : c.psi_display.toFixed(0) + '°';
    $('rpm').textContent = a && a.rpm ? a.rpm.toFixed(0) + ' rpm' : '--';
    $('cut').textContent = !a ? 'mic off' : a.cutting == null ? 'calibrating' : a.cutting ? 'CUTTING' : 'not cutting';
    histPsi.push([t, c.psi_display]); histScore.push([t, c.score]);
    const keep = t - SPAN * 4; histPsi = histPsi.filter(p => p[0] > keep); histScore = histScore.filter(p => p[0] > keep);
    drawTrace($('tr-psi'), histPsi, t, 0, 100, css('--acc'), setup && setup.predicted_psi);
    drawTrace($('tr-score'), histScore, t, -2, 6, css('--warn'), 3);
    const al = $('alarm');
    al.className = 'alarm' + (c.alarm ? ' on' : '');
    al.textContent = c.alarm ? `ALARM: chip build-up / tangle since t = ${c.alarm_t.toFixed(1)} s. Stop and clear chips.` : 'chip flow normal';
    const W = { orientation_changed: 'Phone was rotated after calibration: recalibrate.', tracking_lost: 'Tool not found in the picture.', frame_gap: 'Frames are being dropped.' };
    $('warn').textContent = (j.warnings || []).map(w => W[w] || w).join(' ');
  }

  function drawTrace(canvas, pts, now, lo, hi, col, ref) {
    const dpr = devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w) return;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const x = canvas.getContext('2d'); x.scale(dpr, dpr);
    x.strokeStyle = css('--line'); x.strokeRect(0.5, 0.5, w - 1, h - 1);
    const Y = v => h - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * (h - 4) - 2;
    if (ref != null) { x.strokeStyle = css('--pred'); x.setLineDash([4, 4]); x.beginPath(); x.moveTo(0, Y(ref)); x.lineTo(w, Y(ref)); x.stroke(); x.setLineDash([]); }
    x.strokeStyle = col; x.lineWidth = 2; x.beginPath(); let pen = false;
    for (const [t, v] of pts) {
      if (v == null || t < now - SPAN) { pen = false; continue; }
      const px = (t - (now - SPAN)) / SPAN * w;
      pen ? x.lineTo(px, Y(v)) : x.moveTo(px, Y(v)); pen = true;
    }
    x.stroke();
  }

  // ------------------------------------------------------------ video + overlays
  function arrow(x, y, len, d, col, w) {
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = w; g.beginPath(); g.moveTo(x, y);
    const ex = x + len * d[0], ey = y + len * d[1]; g.lineTo(ex, ey); g.stroke();
    const a = Math.atan2(ey - y, ex - x); g.beginPath(); g.moveTo(ex, ey);
    g.lineTo(ex - 14 * Math.cos(a - 0.4), ey - 14 * Math.sin(a - 0.4)); g.lineTo(ex - 14 * Math.cos(a + 0.4), ey - 14 * Math.sin(a + 0.4)); g.fill();
  }
  function draw() {
    if (img) g.drawImage(img, 0, 0); else { g.fillStyle = '#0e0e10'; g.fillRect(0, 0, cv.width, cv.height); }
    if (step === 2) return drawCalibration();
    const c = last && last.chip;
    if (!img || !c || !c.tip || !c.valid) return;
    const [tx, ty] = c.tip, uf = c.u_f, uo = c.u_o;
    g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.arc(tx, ty, 25, 0, 6.283); g.stroke();
    arrow(tx, ty, 70, uf, '#3da0ff', 3);
    if (c.psi_display != null) {
      const a = c.psi_display * Math.PI / 180;
      arrow(tx, ty, 200, [Math.cos(a) * uf[0] + Math.sin(a) * uo[0], Math.cos(a) * uf[1] + Math.sin(a) * uo[1]], '#ffd23d', 4);
    }
  }
  const P = q => [q[0] * cv.width, q[1] * cv.height];
  function drawCalibration() {
    const mark = (q, col, label) => { if (!q) return; const [x, y] = P(q); g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2;
      g.beginPath(); g.arc(x, y, 10, 0, 6.283); g.stroke(); g.font = '16px system-ui'; g.fillText(label, x + 14, y - 8); };
    mark(pts.tip, '#ffd23d', 'tip'); mark(pts.feed, '#3da0ff', 'feed'); mark(pts.work, '#4cc16a', 'workpiece');
    mark(pts.s1, '#fff', ''); mark(pts.s2, '#fff', 'scale');
    if (pts.tip && pts.feed) { const [x, y] = P(pts.tip), [x2, y2] = P(pts.feed); g.strokeStyle = '#3da0ff'; g.beginPath(); g.moveTo(x, y); g.lineTo(x2, y2); g.stroke(); }
    if (pts.s1 && pts.s2) { const [x, y] = P(pts.s1), [x2, y2] = P(pts.s2); g.strokeStyle = '#fff'; g.beginPath(); g.moveTo(x, y); g.lineTo(x2, y2); g.stroke(); }
  }

  // ------------------------------------------------------------ calibration
  const order = ['tip', 'feed', 'work', 's1', 's2'];
  function refreshChecks() {
    $('k-tip').classList.toggle('done', !!pts.tip); $('k-feed').classList.toggle('done', !!pts.feed);
    $('k-work').classList.toggle('done', !!pts.work); $('k-scale').classList.toggle('done', !!(pts.s1 && pts.s2));
    $('k-idle').classList.toggle('done', idleOk);
    ['tip', 'feed', 'work', 'scale'].forEach(k => $('k-' + k).classList.toggle('active', tool === (k === 'scale' ? (pts.s1 ? 's2' : 's1') : k) || (k === 'scale' && (tool === 's1' || tool === 's2'))));
    $('apply').disabled = !(pts.tip && pts.feed && pts.work && status.phone === 'connected');
    $('to-live').disabled = !applied;
  }
  document.querySelectorAll('[data-tool]').forEach(b => b.onclick = () => {
    const t = b.dataset.tool; tool = t === 'scale' ? 's1' : t;
    if (t === 'scale') { pts.s1 = pts.s2 = null; }
    refreshChecks();
  });
  cv.onclick = e => {
    if (step !== 2 || !tool) return;
    const r = cv.getBoundingClientRect(), q = [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
    pts[tool] = q;
    tool = tool === 's1' ? 's2' : null; applied = false;
    refreshChecks(); draw();
  };
  $('apply').onclick = async () => {
    const mm = parseFloat($('in-scale').value);
    const body = { tip: pts.tip, feed_pt: pts.feed, work_pt: pts.work, scale_pts: pts.s1 && pts.s2 ? [pts.s1, pts.s2] : null, scale_mm: mm > 0 ? mm : null };
    const r = await post('/api/calibrate', body), j = await r.json();
    if (!j.ok) { $('cal-msg').textContent = j.error; $('cal-msg').style.color = css('--bad'); return; }
    applied = true; histPsi = []; histScore = [];
    const W = { tip_near_edge: 'The tool will leave the picture during the pass: move the phone back or aim along the feed direction.' };
    $('cal-msg').style.color = j.warnings.length ? css('--warn') : css('--ok');
    $('cal-msg').textContent = j.warnings.length ? j.warnings.map(w => W[w] || w).join(' ') : 'Calibration applied.' + (j.mm_per_px ? ` Scale: ${(1 / j.mm_per_px).toFixed(1)} px/mm.` : '');
    refreshChecks();
  };
  $('idle-btn').onclick = async () => {
    $('idle-txt').textContent = 'recording... keep the spindle running, no cutting';
    await new Promise(r => setTimeout(r, 3000));
    const r = await post('/api/idle_noise', setup ? setup.params : {}), j = await r.json();
    idleOk = !!j.ok;
    $('idle-txt').style.color = j.ok ? css('--ok') : css('--warn');
    $('idle-txt').textContent = j.error ? j.error : `heard ${j.rpm_detected.toFixed(0)} rpm, expected ${j.rpm_expected.toFixed(0)}` + (j.ok ? '' : ': does not match, check the diameter / speed');
    refreshChecks();
  };
  $('to-live').onclick = () => go(3);

  // ------------------------------------------------------------ setup + summary
  let timer;
  async function sendSetup() {
    const body = { d_mm: +$('in-d').value, f: +$('in-f').value, ap: +$('in-ap').value, vc: +$('in-vc').value, r_eps: +$('in-re').value, material: $('in-mat').value };
    setup = await (await post('/api/setup', body)).json();
    $('s-pred').textContent = setup.predicted_psi.toFixed(0) + '°'; $('pred').textContent = setup.predicted_psi.toFixed(0) + '°';
    $('s-rpm').textContent = setup.rpm_expected.toFixed(0) + ' rpm';
    $('s-extrap').hidden = !setup.extrapolation;
    const th = await (await fetch(`/api/theory?f=${setup.params.f}&ap=${setup.params.ap}&r_eps=${setup.params.r_eps}`)).json();
    $('s-theory').textContent = th.error ? '--' : `${th.psi_theory_colwell.toFixed(0)}°  (η = ${th.eta_colwell.toFixed(0)}°; Young-Mathew-Oxley ${th.psi_theory_ymo.toFixed(0)}°)`;
    if (forms) window.drawChipChart($('chart'), forms, setup.params);
  }
  ['in-d', 'in-f', 'in-ap', 'in-vc', 'in-re', 'in-mat'].forEach(id => $(id).oninput = () => { clearTimeout(timer); timer = setTimeout(sendSetup, 200); });
  async function loadSummary() {
    const s = await (await fetch('/api/summary')).json();
    const f = (v, d = 0, u = '') => v == null ? '--' : v.toFixed(d) + u;
    $('sum').innerHTML = [
      ['measured chip angle', f(s.measured_psi_median, 0, '°') + (s.measured_psi_iqr ? ` (middle half ${f(s.measured_psi_iqr[0])}-${f(s.measured_psi_iqr[1])}°)` : ''), 'm', 'measured'],
      ['our model predicted', f(s.predicted_psi, 0, '°'), 'o', 'our model'],
      ['difference', f(s.error_deg, 1, '°'), '', ''], ['frames used', `${s.valid_frames} of ${s.frames}`, '', ''],
      ['cut length in time', f(s.cut_duration_s, 1, ' s'), '', ''], ['build-up alarm', s.alarm_t == null ? 'none' : `at t = ${f(s.alarm_t, 1)} s`, '', ''],
    ].map(r => `<div><div class="k">${r[0]}${r[3] ? ` <span class="tag ${r[2]}">${r[3]}</span>` : ''}</div><div class="v">${r[1]}</div></div>`).join('');
    $('sum-note').textContent = s.extrapolation ? 'These settings are outside the six tested runs, so the model value is an extrapolation.' : 'The model was fitted on six runs; its leave-one-run-out error is about 6.6°.';
  }

  async function init() {
    $('qr').src = '/api/qr?' + Date.now();
    const cfg = await (await fetch('/api/config')).json();
    $('url').textContent = cfg.phone_url;
    for (const r of cfg.runs) {
      const b = document.createElement('button'); b.textContent = r;
      b.onclick = () => { histPsi = []; histScore = []; post('/api/replay/' + r); };
      $('runs').appendChild(b);
    }
    document.querySelectorAll('#nav button').forEach(b => b.onclick = () => go(+b.dataset.step));
    $('to-cal').onclick = () => go(2); $('to-sum').onclick = () => go(4);
    $('stop').onclick = () => post('/api/stop');
    forms = await (await fetch('/api/chip_forms')).json();
    await sendSetup(); refreshChecks(); go(1); connect();
  }
  init();
})();
