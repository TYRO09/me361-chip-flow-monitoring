// Phone page: streams camera frames (JPEG, long side 960 px) and microphone audio (16 kHz mono float32) to the laptop.
(() => {
  const $ = id => document.getElementById(id);
  const LONG = 960, QUALITY = 0.7, MAX_INFLIGHT = 2;
  let ws, stream, running = false, inflight = [], sent = 0, rtts = [], ctx, wake;
  const say = (id, txt, cls) => { const e = $(id); e.textContent = txt; e.className = cls || ''; };

  // Area-averaging resampler to 16 kHz: acts as the anti-alias filter, so no hiss folds into the 2-6 kHz band.
  const workletSrc = `
  class Rec extends AudioWorkletProcessor {
    constructor() { super(); this.r = sampleRate / 16000; this.need = this.r; this.acc = 0; this.out = []; }
    process(inputs) {
      const x = inputs[0][0]; if (!x) return true;
      for (let i = 0; i < x.length; i++) {
        const s = x[i];
        if (this.need > 1 + 1e-9) { this.acc += s; this.need -= 1; }
        else { const w = this.need; this.acc += s * w; this.out.push(this.acc / this.r); this.acc = s * (1 - w); this.need = this.r - (1 - w); }
      }
      if (this.out.length >= 1600) { const f = new Float32Array(this.out); this.out = []; this.port.postMessage(f, [f.buffer]); }
      return true;
    }
  }
  registerProcessor('rec', Rec);`;

  async function start() {
    $('msg').textContent = '';
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } catch (e) {
      try { stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } }); say('s-mic', 'denied (video only)', 'bad'); }
      catch (e2) { $('msg').textContent = 'Camera not available: ' + e2.message; return; }
    }
    const v = $('v'); v.srcObject = stream; await v.play();
    say('s-cam', v.videoWidth + 'x' + v.videoHeight, 'ok');
    const hasAudio = stream.getAudioTracks().length > 0;
    connect(hasAudio);
    if (hasAudio) await startAudio();
    if (window.DeviceOrientationEvent) {
      try { if (DeviceOrientationEvent.requestPermission) await DeviceOrientationEvent.requestPermission(); } catch (e) {}
      let last = 0;
      addEventListener('deviceorientation', ev => {
        const n = performance.now(); if (n - last < 200) return; last = n;
        if (ev.beta == null) return;
        say('s-tilt', `${ev.beta.toFixed(0)}° / ${ev.gamma.toFixed(0)}°`);
        if (ws && ws.readyState === 1) ws.send(JSON.stringify({ tilt: { beta: ev.beta, gamma: ev.gamma } }));
      });
    }
    try { wake = await navigator.wakeLock.request('screen'); } catch (e) {}
    running = true; $('go').textContent = 'Stop'; $('go').className = 'stop';
    loop();
  }

  function connect(hasAudio) {
    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/phone`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      say('s-link', 'connected', 'ok');
      ws.send(JSON.stringify({ hello: 'phone', audio_sr: hasAudio ? 16000 : null, fps: 30, rtt_ms: 0 }));
    };
    ws.onmessage = ev => {
      const j = JSON.parse(ev.data);
      if (j.type === 'ack') { const t0 = inflight.shift(); if (t0) { rtts.push(performance.now() - t0); if (rtts.length > 30) rtts.shift(); } }
      if (j.type === 'refused') { $('msg').textContent = 'Refused: ' + j.reason; say('s-link', 'refused', 'bad'); stop(); }
    };
    ws.onclose = () => { say('s-link', 'disconnected', 'bad'); if (running) setTimeout(() => { if (running) connect(hasAudio); }, 1000); };
    setInterval(() => {
      if (ws.readyState === 1 && rtts.length) ws.send(JSON.stringify({ rtt_ms: rtts.reduce((a, b) => a + b) / rtts.length }));
      say('s-fps', sent + ' fps'); sent = 0;
    }, 1000);
  }

  async function startAudio() {
    try {
      ctx = new AudioContext({ sampleRate: 16000 });
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([workletSrc], { type: 'text/javascript' })));
      const src = ctx.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
      const node = new AudioWorkletNode(ctx, 'rec');
      node.port.onmessage = ev => {
        if (!ws || ws.readyState !== 1) return;
        const f = ev.data, buf = new Uint8Array(1 + f.byteLength);
        buf[0] = 2; buf.set(new Uint8Array(f.buffer), 1); ws.send(buf);
      };
      src.connect(node);
      say('s-mic', 'on, ' + ctx.sampleRate + ' Hz', 'ok');
    } catch (e) { say('s-mic', 'failed: ' + e.message, 'bad'); }
  }

  function loop() {
    const v = $('v'), c = document.createElement('canvas');
    const w = v.videoWidth, h = v.videoHeight, s = LONG / Math.max(w, h);
    c.width = Math.round(w * s); c.height = Math.round(h * s);
    const g = c.getContext('2d');
    const tick = () => {
      if (!running) return;
      if (ws && ws.readyState === 1 && inflight.length < MAX_INFLIGHT && ws.bufferedAmount < 600000) {
        g.drawImage(v, 0, 0, c.width, c.height);
        const t = performance.now();
        c.toBlob(async b => {
          if (!b || ws.readyState !== 1) return;
          const body = new Uint8Array(await b.arrayBuffer()), buf = new Uint8Array(9 + body.length);
          buf[0] = 1; new DataView(buf.buffer).setFloat64(1, t, false); buf.set(body, 9);
          inflight.push(performance.now()); ws.send(buf); sent++;
        }, 'image/jpeg', QUALITY);
      }
      if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(tick); else setTimeout(tick, 33);
    };
    tick();
  }

  function stop() {
    running = false; $('go').textContent = 'Start camera and microphone'; $('go').className = '';
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (ws) ws.close(); if (ctx) ctx.close(); if (wake) wake.release();
    say('s-cam', 'off'); say('s-mic', 'off');
  }
  $('go').onclick = () => running ? stop() : start();
})();
