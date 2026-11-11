"""Streamlit app for the ME361 chip-flow monitor.

Run with:  uv run streamlit run app.py

Upload a top-view clip, click the tool tip and directions, run the full pipeline (code/pipeline.py)
and get the dashboard video, graphs and numbers.
"""
import glob
import hashlib
import json
import os
import re
import sys

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(ROOT, "uploads")
ALARM_THR = 3.0  # same threshold as code/entangle_indicator.py

sys.path.insert(0, os.path.join(ROOT, "code"))

st.set_page_config(page_title="ME361 chip flow monitor", layout="wide")


# ---------- shared helpers ----------
@st.cache_data
def video_bytes(path, mtime=None):
    with open(path, "rb") as fh:
        return fh.read()


SPEEDS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

PLAYER_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
 body{margin:0;font-family:system-ui,-apple-system,sans-serif;font-size:14px;color:#31333f;background:transparent}
 video{width:100%;max-height:640px;background:#1c1c1e;border-radius:6px;display:block}
 .bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:8px}
 button{border:1px solid #d0d0d8;background:#fff;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:13px}
 button.on{background:#ff4b4b;border-color:#ff4b4b;color:#fff}
 .lab{margin-right:4px;color:#6b6b78} .sep{width:1px;height:22px;background:#d0d0d8;margin:0 4px}
 .t{margin-left:auto;font-variant-numeric:tabular-nums;color:#6b6b78}
 @media (prefers-color-scheme: dark){body{color:#fafafa}button{background:#262730;color:#fafafa;border-color:#4a4a55}}
</style></head><body>
<video id="v" src="data:video/mp4;base64,__B64__" controls playsinline preload="auto"></video>
<div class="bar"><span class="lab">Speed</span>__BUTTONS__<span class="sep"></span>
 <button id="back" title="one frame back (,)">◀ frame</button><button id="fwd" title="one frame forward (.)">frame ▶</button>
 <span class="t" id="t"></span></div>
<script>
 const v=document.getElementById('v'), FPS=__FPS__;
 const bs=[...document.querySelectorAll('button[data-r]')];
 function setRate(r){v.playbackRate=r;v.defaultPlaybackRate=r;bs.forEach(b=>b.classList.toggle('on',+b.dataset.r===r));}
 bs.forEach(b=>b.onclick=()=>setRate(+b.dataset.r));
 function step(n){v.pause();v.currentTime=Math.max(0,Math.min(v.duration||0,v.currentTime+n/FPS));}
 document.getElementById('back').onclick=()=>step(-1); document.getElementById('fwd').onclick=()=>step(1);
 document.addEventListener('keydown',e=>{if(e.key===',')step(-1);if(e.key==='.')step(1);
   if(e.key===' '){e.preventDefault();v.paused?v.play():v.pause();}});
 const t=document.getElementById('t');
 function upd(){t.textContent=v.currentTime.toFixed(2)+' / '+(v.duration||0).toFixed(1)+' s  ·  '+v.playbackRate+'×';}
 v.addEventListener('timeupdate',upd);v.addEventListener('ratechange',upd);v.addEventListener('loadedmetadata',upd);
 setRate(1);
</script></body></html>"""


def speed_player(path, fps=30.0):
    """Video player with playback-speed buttons, frame stepping and keyboard shortcuts
    (space = play/pause, , and . = one frame back/forward). The video is our own rendered mp4."""
    import base64
    b64 = base64.b64encode(video_bytes(path, os.path.getmtime(path))).decode("ascii")
    buttons = "".join(f'<button data-r="{r}">{r:g}×</button>' for r in SPEEDS)
    html = PLAYER_HTML.replace("__B64__", b64).replace("__BUTTONS__", buttons).replace("__FPS__", f"{fps:.3f}")
    st.iframe(html, height="content")


def angle_chart(d):
    d = d.copy()
    d["psi_smooth"] = d.psi_peak.rolling(15, center=True, min_periods=5).median()  # 0.5 s
    base = alt.Chart(d).encode(x=alt.X("t:Q", title="time in clip (s)"))
    raw = base.mark_circle(size=8, opacity=0.25).encode(y=alt.Y("psi_peak:Q", title="chip angle ψ (°)"))
    smooth = base.mark_line(color="#d62728").encode(y="psi_smooth:Q")
    st.markdown("**Chip ejection angle** (dots = each frame, line = 0.5 s median)")
    st.altair_chart((raw + smooth).properties(height=220), width="stretch")


def score_chart(e, alarm_t):
    st.markdown(f"**Chip build-up score** (alarm when it stays above {ALARM_THR:g} for 1 s)")
    sc = alt.Chart(e).mark_line().encode(x=alt.X("t:Q", title="time in clip (s)"),
                                         y=alt.Y("score_s:Q", title="score (robust z)"))
    layers = sc + alt.Chart(pd.DataFrame({"y": [ALARM_THR]})).mark_rule(strokeDash=[4, 4], color="black").encode(y="y:Q")
    if alarm_t is not None and pd.notna(alarm_t):
        layers += alt.Chart(pd.DataFrame({"t": [alarm_t]})).mark_rule(color="red").encode(x="t:Q")
    st.altair_chart(layers.properties(height=180), width="stretch")


def overlays(paths, strip_prefix):
    if not paths:
        return
    st.subheader("What the measurement sees")
    st.caption("Red = moving chip pixels · blue arrow = feed direction · green arrow = measured chip direction · "
               "cyan rings = measuring annulus")
    cols = st.columns(len(paths))
    for c, p in zip(cols, paths):
        c.image(p, caption=os.path.basename(p)[strip_prefix:-4], width="stretch")


# ---------- new-video pipeline ----------
POINTS = [("tip", "Tool tip", "the cutting corner of the insert, where the chips come off"),
          ("feed", "Feed direction", "any point further along the bar in the direction the tool travels (towards the chuck)"),
          ("out", "Outward side", "any point on the side of the bar where the tool is, away from the workpiece")]


@st.cache_data
def cached_info(path, mtime):
    import pipeline as P
    return P.video_info(path)


@st.cache_data
def cached_frame(path, mtime, t):
    import pipeline as P
    return P.read_frame(path, t)


def save_upload(up):
    data = up.getvalue()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(up.name)[0])[:40] or "clip"
    folder = os.path.join(UPLOADS, f"{stem}_{hashlib.sha1(data).hexdigest()[:8]}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "video" + os.path.splitext(up.name)[1].lower())
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(data)
    return folder, path


def pick_points(path, mtime, ref_t):
    """Clickable reference frame; returns dict of picked points in 960-px working coordinates."""
    import pipeline as P
    ss = st.session_state
    pts = ss.setdefault("pts", {})
    if ss.get("pts_for") != (path, ref_t):          # new clip or new reference frame -> start over
        pts.clear(); ss["pts_for"] = (path, ref_t); ss["last_click"] = None
    nxt = next((k for k, _, _ in POINTS if k not in pts), None)

    c1, c2 = st.columns([1.6, 1])
    with c2:
        st.markdown("**Click on the frame, in this order:**")
        for k, name, desc in POINTS:
            mark = "✅" if k in pts else ("👉" if k == nxt else "⬜")
            st.markdown(f"{mark} **{name}**: {desc}")
        st.caption("Blue arrow = feed (ψ = 0°), green arrow = outward (ψ = 90°). "
                   "Pink box = the patch of the insert the tracker follows; it should sit on the insert, "
                   "not on chips or background. Yellow rings = where chips are measured (0/45/90 = ψ in degrees).")
        b1, b2 = st.columns(2)
        if b1.button("Undo last point", disabled=not pts):
            pts.pop(next(k for k, _, _ in reversed(POINTS) if k in pts)); st.rerun()
        if b2.button("Clear points", disabled=not pts):
            pts.clear(); st.rerun()
    with c1:
        img = cached_frame(path, mtime, ref_t)
        shown = P.draw_setup(img, pts.get("tip"), pts.get("feed"), pts.get("out"))
        click = streamlit_image_coordinates(shown, key=f"pick_{hash((path, ref_t))}", width=min(img.shape[1], 760),
                                            cursor="crosshair")
        if click and click.get("unix_time") != ss.get("last_click") and nxt:
            ss["last_click"] = click.get("unix_time")
            s = img.shape[1] / click["width"]
            pts[nxt] = (round(click["x"] * s, 1), round(click["y"] * s, 1))
            st.rerun()
    return pts


def show_result(folder):
    with open(os.path.join(folder, "summary.json")) as fh:
        s = json.load(fh)
    st.header("Results")
    if s.get("warning"):
        st.warning(s["warning"])
    m = st.columns(3)
    m[0].metric("Chip angle ψ (median)", f"{s['psi_median']:.0f}°", help=f"IQR {s['psi_iqr'][0]:.0f}–{s['psi_iqr'][1]:.0f}°")
    m[1].metric("Model prediction", f"{s['psi_pred']:.0f}°" if s["psi_pred"] is not None else "—",
                help="ψ ≈ 26.8 + 101·f + 17.7·ap, fitted on the six lab runs")
    m[2].metric("Tangle alarm", f"at {s['alarm_s']:.1f} s" if s["alarm_s"] is not None else "none",
                help=f"max build-up score {s['max_score']} (threshold {ALARM_THR:g})")
    st.caption(f"{s['valid_frames']} of {s['frames']} frames measured · tool tracked in {s['tracking_ok_pct']}% of frames · "
               f"window {s['window'][0]:.1f}–{s['window'][1]:.1f} s")

    left, right = st.columns([1.2, 1])
    with left:
        dash = os.path.join(folder, "dashboard.mp4")
        if os.path.exists(dash):
            st.markdown("**Dashboard video**")
            speed_player(dash, s.get("fps") or 30.0)
            with open(dash, "rb") as fh:
                st.download_button("Download dashboard video", fh, file_name="dashboard.mp4", mime="video/mp4")
    with right:
        fr = pd.read_csv(os.path.join(folder, "frames.csv")).query("valid == 1")
        angle_chart(fr)
        ent = pd.read_csv(os.path.join(folder, "entangle.csv"))
        if not ent.empty:
            score_chart(ent, s["alarm_s"])
        with open(os.path.join(folder, "frames.csv"), "rb") as fh:
            st.download_button("Download per-frame data (CSV)", fh, file_name="frames.csv", mime="text/csv")
    overlays(sorted(glob.glob(os.path.join(folder, "overlays", "U_t*.jpg"))), 3)


def view_new_video():
    import pipeline as P
    st.caption("Top-view clip of an OD turning cut, like the lab runs: camera looking down on the insert, "
               "chips visible. The clip is saved under uploads/ and analysed locally.")
    up = st.file_uploader("Video file", type=["mp4", "mov", "m4v", "avi", "mkv"])
    if up is None:
        return
    folder, path = save_upload(up)
    mtime = os.path.getmtime(path)
    try:
        info = cached_info(path, mtime)
    except ValueError as e:
        st.error(str(e)); return
    dur = info["duration"]

    with st.sidebar:
        st.subheader("Cut settings (optional)")
        f = st.number_input("Feed f (mm/rev)", 0.0, 1.0, 0.0, 0.01, format="%.3f",
                            help="With f and ap the dashboard also shows the model's predicted angle")
        ap = st.number_input("Depth of cut ap (mm)", 0.0, 5.0, 0.0, 0.1)
        render = st.checkbox("Render dashboard video", True, help="Takes about as long as the clip itself")

    st.markdown(f"**{up.name}** · {dur:.1f} s · {info['fps']:.0f} fps · {info['width']}×{info['height']}")

    # default window: the whole clip, capped at the memory limit
    w0, w1 = 0.0, min(dur, P.MAX_WINDOW_S)

    st.subheader("1 · Analysis window")
    window = st.slider("Part of the clip to analyse (s): pick steady cutting only. The first 4 s of the window "
                       "are the 'normal chip flow' baseline for the tangle alarm.", 0.0, float(round(dur, 1)),
                       (float(round(w0, 1)), float(round(w1, 1))), 0.1)
    if window[1] - window[0] > P.MAX_WINDOW_S:
        st.error(f"Keep the window to {P.MAX_WINDOW_S:.0f} s or less (all frames are held in memory).")
        return

    st.subheader("2 · Mark the tool on a reference frame")
    ref_t = st.slider("Reference frame (s): choose one where the insert is clearly visible",
                      float(window[0]), float(window[1]), float(round((window[0] + window[1]) / 2, 1)), 0.1)
    pts = pick_points(path, mtime, ref_t)
    ready = all(k in pts for k, _, _ in POINTS)

    st.subheader("3 · Run")
    err = None
    if ready:
        try:
            fd, od = P.directions(pts["tip"], pts["feed"], pts["out"])
            warn = P.check_setup(info, pts["tip"], fd, od, ref_t, window)
            if warn:
                st.warning(warn)
        except ValueError as e:
            err = str(e); st.error(err)
    if st.button("Run analysis", type="primary", disabled=not ready or err is not None):
        labels = {"measure": "Measuring chip angle frame by frame"}
        if render:
            labels["render"] = "Rendering dashboard video"
        bars = {k: st.progress(0.0, text=t) for k, t in labels.items()}
        order = list(bars)

        def prog(stage, frac):
            if stage not in bars:
                return
            for k in order[:order.index(stage)]:      # earlier stages are finished
                bars[k].progress(1.0, text=labels[k])
            bars[stage].progress(min(1.0, frac), text=labels[stage])
        try:
            P.analyze(path, pts["tip"], pts["feed"], pts["out"], ref_t, window, folder,
                      f=f or None, ap=ap or None, render=render, progress=prog, use_audio=False)
            st.session_state["result_for"] = folder
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            st.error(f"Analysis failed: {e}")
        for b in bars.values():
            b.empty()
    elif not ready:
        st.info("Mark all three points to enable the analysis.")

    if st.session_state.get("result_for") == folder and os.path.exists(os.path.join(folder, "summary.json")):
        show_result(folder)


view_new_video()
