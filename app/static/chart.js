// f-ap chart in the axis format of ISO 10910 (feed on x, depth of cut on y, both logarithmic).
// Only our six observed runs and the current setting are drawn. Chip-class boundaries depend on the insert and the
// material; we have no data for them, so none are drawn.
window.drawChipChart = function (canvas, forms, current) {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const dpr = devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext('2d'); g.scale(dpr, dpr);
  const L = 56, R = 12, T = 10, B = 34, [f0, f1] = forms.axes.f, [a0, a1] = forms.axes.ap;
  const X = f => L + (Math.log(f) - Math.log(f0)) / (Math.log(f1) - Math.log(f0)) * (w - L - R);
  const Y = a => h - B - (Math.log(a) - Math.log(a0)) / (Math.log(a1) - Math.log(a0)) * (h - T - B);
  g.font = '12px system-ui'; g.textBaseline = 'middle';
  g.strokeStyle = css('--line'); g.fillStyle = css('--mut');
  g.strokeRect(L, T, w - L - R, h - T - B);
  for (const f of [0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5]) { g.beginPath(); g.moveTo(X(f), T); g.lineTo(X(f), h - B); g.stroke(); g.textAlign = 'center'; g.fillText(f, X(f), h - B + 12); }
  for (const a of [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]) { g.beginPath(); g.moveTo(L, Y(a)); g.lineTo(w - R, Y(a)); g.stroke(); g.textAlign = 'right'; g.fillText(a, L - 6, Y(a)); }
  g.textAlign = 'center'; g.fillText('feed f (mm/rev)', (L + w - R) / 2, h - 8);
  g.save(); g.translate(10, (T + h - B) / 2); g.rotate(-Math.PI / 2); g.fillText('depth of cut ap (mm)', 0, 0); g.restore();
  for (const r of forms.runs) {
    const x = X(r.f), y = Y(r.ap);
    g.fillStyle = r.tangled ? css('--bad') : css('--acc'); g.beginPath(); g.arc(x, y, 6, 0, 6.283); g.fill();
    g.fillStyle = css('--fg'); g.textAlign = 'left'; g.fillText(r.run, x + 9, y);
  }
  if (current && current.f > 0 && current.ap > 0) {
    const x = X(Math.min(Math.max(current.f, f0), f1)), y = Y(Math.min(Math.max(current.ap, a0), a1));
    g.strokeStyle = css('--pred'); g.lineWidth = 2; g.beginPath(); g.moveTo(x, y - 9); g.lineTo(x + 9, y); g.lineTo(x, y + 9); g.lineTo(x - 9, y); g.closePath(); g.stroke(); g.lineWidth = 1;
  }
};
