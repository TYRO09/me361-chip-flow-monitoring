// Dashboard: websocket client + drawing. Everything shown comes from the server; nothing is computed in the page
// except the display layout.
(() => {
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const cv = $('cv'), g = cv.getContext('2d');
  let img = null, last = null, histPsi = [], histScore = [], predicted = null, theory = null;
  const SPAN = 8;                                  // seconds shown in the traces

  function chip(id, txt, cls) { const e = $(id); e.textContent = txt; e.className = 'chip ' + (cls || ''); }

  function connect() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/dash`);
    ws.binaryType = 'blob';
    ws.onmessage = async ev => {
      if (typeof ev.data === 'string') return onJson(JSON.parse(ev.data));
      const blob = ev.data.slice(1, ev.data.size, 'image/jpeg');
      const bmp = await createImageBitmap(blob);
      if (cv.width !== bmp.width) { cv.width = bmp.width; cv.height = bmp.height; }
      img = bmp; $('ph').hidden = true; draw();
    };
    ws.onclose = () => setTimeout(connect, 1000);
  }

  function onJson(j) {
    if (j.type === 'status') {
      chip('c-phone', 'phone: ' + j.phone, j.phone === 'connected' ? 'ok' : '');
      chip('c-mic', 'mic: ' + (j.mic ? 'on' : 'off'), j.mic ? 'ok' : 'warn');
      chip('c-cal', j.calibrated ? 'calibrated' : 'not calibrated', j.calibrated ? 'ok' : '');
      $('b-replay').hidden = j.mode !== 'replay';
    } else if (j.type === 'state') {
      last = j; onState(j); draw();
    }
  }

  function onState(j) {
    const c = j.chip, a = j.audio, t = j.t;
    chip('c-fps', j.fps.toFixed(0) + ' fps', j.fps >= 20 ? 'ok' : 'warn');
    chip('c-lat', 'latency ' + j.latency_ms.toFixed(0) + ' ms', j.latency_ms < 300 ? 'ok' : 'warn');
    if (j.mode === 'replay') $('b-replay').hidden = false;
    $('psi').textContent = c.psi_display == null ? '--' : c.psi_display.toFixed(0) + '°';
    $('rpm').textContent = a && a.rpm ? a.rpm.toFixed(0) + ' rpm' : '--';
    $('cut').textContent = !a ? 'mic off' : a.cutting == null ? 'calibrating' : a.cutting ? 'CUTTING' : 'not cutting';
    histPsi.push([t, c.psi_display]); histScore.push([t, c.score]);
    const cut = t - SPAN * 4; histPsi = histPsi.filter(p => p[0] > cut); histScore = histScore.filter(p => p[0] > cut);
    drawTrace($('tr-psi'), histPsi, t, 0, 100, css('--acc'), predicted);
    drawTrace($('tr-score'), histScore, t, -2, 6, css('--warn'), 3);
    const al = $('alarm');
    al.className = 'alarm' + (c.alarm ? ' on' : '');
    al.textContent = c.alarm ? `ALARM: chip build-up / tangle since t = ${c.alarm_t.toFixed(1)} s. Stop and clear chips.` : 'chip flow normal';
  }

  function drawTrace(canvas, pts, now, lo, hi, col, ref) {
    const dpr = devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
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

  function draw() {
    if (!img) return;
    g.drawImage(img, 0, 0);
    const c = last && last.chip;
    if (!c || !c.tip || !c.valid) return;
    const [tx, ty] = c.tip, uf = c.u_f, uo = c.u_o;
    const arrow = (len, ang, col, w) => {
      const d = [Math.cos(ang) * uf[0] + Math.sin(ang) * uo[0], Math.cos(ang) * uf[1] + Math.sin(ang) * uo[1]];
      g.strokeStyle = col; g.fillStyle = col; g.lineWidth = w; g.beginPath(); g.moveTo(tx, ty);
      const ex = tx + len * d[0], ey = ty + len * d[1]; g.lineTo(ex, ey); g.stroke();
      const a = Math.atan2(ey - ty, ex - tx); g.beginPath(); g.moveTo(ex, ey);
      g.lineTo(ex - 14 * Math.cos(a - 0.4), ey - 14 * Math.sin(a - 0.4)); g.lineTo(ex - 14 * Math.cos(a + 0.4), ey - 14 * Math.sin(a + 0.4)); g.fill();
    };
    g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.arc(tx, ty, 25, 0, 6.283); g.stroke();
    arrow(70, 0, '#3da0ff', 3);
    if (c.psi_display != null) arrow(200, c.psi_display * Math.PI / 180, '#ffd23d', 4);
  }

  async function init() {
    $('qr').src = '/api/qr?' + Date.now();
    const cfg = await (await fetch('/api/config')).json();
    $('url').textContent = cfg.phone_url;
    for (const r of cfg.runs) {
      const b = document.createElement('button'); b.textContent = r;
      b.onclick = () => { histPsi = []; histScore = []; fetch('/api/replay/' + r, { method: 'POST' }); };
      $('runs').appendChild(b);
    }
    $('stop').onclick = () => fetch('/api/stop', { method: 'POST' });
    connect();
  }
  init();
})();
