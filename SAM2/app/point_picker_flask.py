#!/usr/bin/env python3
"""
SAM2 / Point Picker mini-app
----------------------------
Single-file Flask server that serves:
  • "/"     : Home (Apple-like UI) to choose 'use existing' or create new prompts
  • "/pick" : Interactive point picker (Apple-like UI)
  • "/frame": Serves frame images by index
  • "/save" : Saves prompts.json with POS/NEG points for frame 0

Design goals:
  • Modular helpers
  • Readable English comments and docstrings
  • Minimal side-effects; safe filesystem operations via pathlib
"""

from __future__ import annotations

import json
import glob
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any


from flask import Flask, request, redirect, render_template_string, send_from_directory, jsonify


# =============================================================================
# Configuration & Environment
# =============================================================================

DATASET_NAME = os.environ.get("DATASET_NAME", "").strip()
INPUT = os.environ.get("INPUT", "/data/in")
OUT_ROOT = os.environ.get("OUT", "/data/out")
INDEX_SUFFIX = os.environ.get("INDEX_SUFFIX", "_indexed")

# Derive dataset folder name. If user passed INPUT=/data/in/<dataset>, just use tail.
DS_NAME = DATASET_NAME or Path(INPUT.rstrip("/")).name

# Output folder: /data/out/<dataset>_indexed
INDEXED_DIR = Path(OUT_ROOT) / f"{DS_NAME}{INDEX_SUFFIX}"
INDEXED_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_JSON = INDEXED_DIR / "prompts.json"
DONE_FLAG = INDEXED_DIR / "__picker_done.flag"
USE_EXISTING = INDEXED_DIR / "__use_existing.flag"

PREVIEW_DIR = INDEXED_DIR / "preview"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HTML Templates (Apple-like UI)
#  - Kept inline for single-file convenience.
#  - If you prefer, split to separate files later and load them with Flask templates.
# =============================================================================

HOME_HTML = """
<!doctype html><meta charset="utf-8">
<title>Point Picker — {{ds}}</title>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>
  :root{
    --bg: #0b0b0d; --ink:#f5f5f7; --muted:#a3a3aa;
    --card: rgba(22,22,24,0.66); --border: rgba(255,255,255,0.08);
    --accent:#0a84ff; --shadow: 0 30px 80px rgba(0,0,0,0.45); --glass: saturate(180%) blur(22px);
  }
  @media (prefers-color-scheme: light){
    :root{ --bg:#f5f5f7; --ink:#0b0b0d; --muted:#6e6e73; --card:rgba(255,255,255,0.72); --border:rgba(0,0,0,0.08); --shadow:0 30px 70px rgba(0,0,0,0.12); }
  }
  *{ box-sizing:border-box }
  html,body{ height:100%; }
  body{
    margin:0; color:var(--ink);
    background:
      radial-gradient(1200px 800px at 10% -10%, rgba(10,132,255,0.14), transparent 60%),
      radial-gradient(1000px 700px at 120% 110%, rgba(94,92,230,0.14), transparent 60%),
      var(--bg);
    font-family: -apple-system, BlinkMacSystemFont,
                 "SF Pro Text","SF Pro Display",
                 "Segoe UI", Roboto, Helvetica, Arial,
                 "Apple Color Emoji","Segoe UI Emoji";
    font-size:15px; line-height:1.55;
    -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; text-rendering: optimizeLegibility;
    display:flex; align-items:center; justify-content:center; padding:24px;
  }

  .wrap{ width:100%; max-width: 920px; display:flex; flex-direction:column; gap:16px; }
  .hero{
    border:1px solid var(--border); border-radius:18px;
    background:var(--card); backdrop-filter:var(--glass); box-shadow:var(--shadow);
    padding:16px 18px; display:flex; align-items:center; gap:12px; position:relative; overflow:hidden;
  }
  .badge{ display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border:1px solid var(--border);
          border-radius:999px; background:rgba(255,255,255,0.06); color:var(--muted); font-weight:600; }
  .dot{ width:8px; height:8px; border-radius:50%; background:var(--accent); box-shadow:0 0 0 5px rgba(10,132,255,0.18); }
  .title{ font-weight:800; letter-spacing:.2px; font-size:18px; }
  .muted{ color:var(--muted); }

  .card{
    border:1px solid var(--border); border-radius:18px;
    background:var(--card); backdrop-filter:var(--glass); box-shadow:var(--shadow);
    padding:16px 18px; display:flex; flex-direction:column; gap:14px;
  }
  .row{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .spacer{ flex:1 1 auto; }

  .btn{
    appearance:none; padding:10px 14px; border-radius:12px;
    border:1px solid rgba(0,0,0,0.08);
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.82));
    color:#111; font-weight:700; letter-spacing:.2px;
    box-shadow:0 10px 22px rgba(0,0,0,0.10);
    transition: transform .06s ease, box-shadow .2s ease, background .2s ease;
    cursor:pointer; text-decoration:none; display:inline-flex; align-items:center; gap:8px;
  }
  .btn:hover{ transform: translateY(-1px); box-shadow:0 12px 26px rgba(0,0,0,0.14); }
  .btn:active{ transform: translateY(0); box-shadow:0 8px 18px rgba(0,0,0,0.10); }
  .btn-primary{ background: linear-gradient(180deg, var(--accent), #0568db); color:#fff; border:none; box-shadow: 0 14px 28px rgba(10,132,255,0.35); }

  .files{
    border:1px solid var(--border); border-radius:18px; background:var(--card); backdrop-filter:var(--glass); box-shadow:var(--shadow);
    padding:14px 16px;
  }
  .files ul{ margin:8px 0 0 18px; padding:0; }
  .files li{ padding:4px 0; color:var(--muted); }

  .pill{ padding:6px 10px; border:1px solid var(--border); border-radius:999px; color:var(--muted); background:rgba(255,255,255,0.06); font-weight:600; }
</style>

<div class="wrap">
  <div class="hero">
    <span class="badge"><span class="dot"></span> Interactive Picker</span>
    <div class="title">Point Picker — <span class="muted">{{ds}}</span></div>
    <div class="spacer"></div>
    <div class="pill">Frames: {{nframes}}</div>
  </div>

  {% if exists %}
  <div class="card">
    <div class="row">
      <div class="title">A prompts.json already exists</div>
      <div class="spacer"></div>
      <form method="post" action="/use_existing">
        <button class="btn" type="submit">Use existing</button>
      </form>
      <form method="post" action="/create_new">
        <button class="btn btn-primary" type="submit">Create new</button>
      </form>
    </div>
    <div class="muted">Choose whether to continue with the saved prompts or start fresh.</div>
  </div>
  {% else %}
  <div class="card">
    <div class="row">
      <div class="title">No prompts yet</div>
      <div class="spacer"></div>
      <form method="get" action="/pick">
        <button class="btn btn-primary" type="submit">Open picker</button>
      </form>
    </div>
    <div class="muted">Start the interactive picker to add POS/NEG points.</div>
  </div>
  {% endif %}

  <div class="files">
    <div class="title" style="font-size:16px;">Files</div>
    <ul>
      <li><strong>PROMPTS_JSON:</strong> {{prompts}}</li>
      <li><strong>DONE_FLAG:</strong> {{done}}</li>
      <li><strong>USE_EXISTING:</strong> {{use_existing}}</li>
    </ul>
  </div>
</div>
"""

PICK_HTML = """
<!doctype html><meta charset="utf-8">
<title>Picker</title>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>
  :root{
    --bg: #0b0b0d; --ink: #f5f5f7; --muted: #a3a3aa;
    --card: rgba(22,22,24,0.66); --border: rgba(255,255,255,0.08);
    --accent: #0a84ff; --pos: #34c759; --neg: #ff3b30;
    --shadow: 0 30px 80px rgba(0,0,0,0.45); --glass: saturate(180%) blur(22px);
  }
  @keyframes bounceDot {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%          { transform: translateY(-4px); opacity: 1; }
  }

  .dot-bounce{
    animation: bounceDot 0.7s ease-in-out infinite;
  }

  @media (prefers-color-scheme: light){
    :root{ --bg:#f5f5f7; --ink:#0b0b0d; --muted:#6e6e73; --card:rgba(255,255,255,0.72); --border:rgba(0,0,0,0.08); --shadow:0 30px 70px rgba(0,0,0,0.12); }
  }
  *{ box-sizing:border-box }
  html,body{ height:100%; }
  body{
    margin:0; color:var(--ink);
    background: radial-gradient(1200px 800px at 10% -10%, rgba(10,132,255,0.14), transparent 60%),
               radial-gradient(1000px 700px at 120% 110%, rgba(94,92,230,0.14), transparent 60%),
               var(--bg);
    font: 15px/1.55 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Helvetica,Arial;
    display:flex; flex-direction:column; gap:12px; padding:12px;
  }

  .shell{ max-width: 1600px; margin:0 auto; width:100%;
          display:grid; grid-template-columns: 1.35fr 0.65fr; gap:12px; }
  .shell.expanded{ grid-template-columns: 1fr; max-width: 96vw; }
  .shell.expanded .drawer{ display:none; }
  .shell.expanded .main{ min-height: 90vh; }

  .hero{
    grid-column: 1 / -1;
    border:1px solid var(--border);
    border-radius:18px; background:var(--card); backdrop-filter:var(--glass); box-shadow:var(--shadow);
    padding:12px 14px; display:flex; align-items:center; gap:14px; overflow:hidden; position:relative;
  }

  .badge{ display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border:1px solid var(--border);
    border-radius:999px; background:rgba(255,255,255,0.06); color:var(--muted); font-weight:600; }
  .dot{ width:8px; height:8px; border-radius:50%; background:var(--accent); box-shadow:0 0 0 5px rgba(10,132,255,0.18); }
  .hero-title{ font-weight:700; letter-spacing:.2px; }
  .hero-sub{ color:var(--muted) }

  .main{
    grid-column:1 / 2;
    border:1px solid var(--border); border-radius:18px; background:var(--card); backdrop-filter:var(--glass); box-shadow:var(--shadow);
    display:flex; flex-direction:column; overflow:hidden; min-height: 86vh;
  }
  .head{ display:flex; align-items:center; gap:12px; padding:10px 12px; border-bottom:1px solid var(--border); }
  .spacer{ flex:1 1 auto; }
  .pill{ padding:6px 10px; border:1px solid var(--border); border-radius:999px; color:var(--muted); background:rgba(255,255,255,0.06); }

  .stage{ position:relative; flex:1 1 auto; min-height:0; overflow:hidden; padding:0;
          background: linear-gradient(180deg, rgba(0,0,0,0.06), transparent 50%), transparent; }

  canvas#c{
    position:absolute; inset:0; margin:0;
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    border-radius:14px; background:#000; border:1px solid var(--border);
    outline: 1px solid rgba(255,255,255,0.04);
    touch-action:none; cursor: crosshair;
  }

  .overlay-ui{
    position:absolute; left:16px; bottom:16px; display:flex; gap:10px; align-items:center; flex-wrap:wrap;
    pointer-events: none;
  }
  .overlay-ui > *{ pointer-events:auto; }

  .btn{
    appearance:none; padding:10px 14px; border-radius:12px;
    border:1px solid rgba(0,0,0,0.08);
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.82));
    color:#111; font-weight:700; letter-spacing:.2px;
    box-shadow:0 10px 22px rgba(0,0,0,0.10);
    transition: transform .06s ease, box-shadow .2s ease, background .2s ease;
    user-select:none;
  }
  .btn:hover{ transform: translateY(-1px); box-shadow:0 12px 26px rgba(0,0,0,0.14); }
  .btn:active{ transform: translateY(0); box-shadow:0 8px 18px rgba(0,0,0,0.10); }
  .btn-ghost{ background: rgba(255,255,255,0.85); color:#111; border:1px solid rgba(0,0,0,0.08); }
  .btn-primary{ background: linear-gradient(180deg, var(--accent), #0568db); color:#fff; border:none; box-shadow: 0 14px 28px rgba(10,132,255,0.35); }

  .seg-switch{ display:inline-flex; gap:6px; align-items:center; padding:6px; border-radius:999px; background:rgba(255,255,255,0.06); border:1px solid var(--border); }
  .seg-switch button{ padding:8px 12px; border-radius:999px; border:1px solid transparent; background:transparent; color:var(--muted); font-weight:700; }
  .seg-switch button.active.pos{ background: var(--pos); color:#fff; }
  .seg-switch button.active.neg{ background: var(--neg); color:#fff; }

  .coords{
    position:absolute; right:16px; bottom:16px; padding:8px 12px; border-radius:10px;
    background: rgba(255,255,255,0.06); border:1px solid var(--border); backdrop-filter:var(--glass); color:var(--muted); font-weight:600;
  }

  .drawer{ grid-column: 2 / 3; width: 300px; display:flex; flex-direction:column; gap:10px; }
  .panel{ border:1px solid var(--border); border-radius:18px; background:var(--card); backdrop-filter:var(--glass); box-shadow:var(--shadow); overflow:hidden; }
  .panel .hd{ padding:12px 14px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:8px; }
  .panel .bd{ padding:10px 12px; max-height: 40vh; overflow:auto; }
  .row{ display:flex; align-items:center; justify-content:space-between; gap:10px; padding:6px 2px; border-bottom:1px dashed rgba(255,255,255,0.06) }
  .row:last-child{ border-bottom:0 }

  .toast{
    position: fixed; left:50%; bottom:22px; transform: translateX(-50%);
    padding:10px 14px; border-radius:12px; border:1px solid var(--border);
    background: var(--card); color:var(--ink); backdrop-filter:var(--glass); box-shadow: var(--shadow);
    opacity:0; pointer-events:none; transition: opacity .25s ease;
  }
  .toast.show{ opacity: 1; }

  .key{ display:inline-flex; align-items:center; justify-content:center; min-width:22px; height:22px; padding:0 6px;
        border:1px solid var(--border); border-radius:8px; background:rgba(255,255,255,0.08); color:var(--muted); font-weight:700; font-size:12px; }

  .crosshair{ position:absolute; inset:0; pointer-events:none }
  .crosshair .h, .crosshair .v{ position:absolute; background:rgba(255,255,255,0.28); }
  .crosshair .h{ height:1px; width:100%; top:0; transform: translateY(var(--y,0)); }
  .crosshair .v{ width:1px; height:100%; left:0; transform: translateX(var(--x,0)); }

  .grid{ position:absolute; inset:0; pointer-events:none;
    background-image: linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px);
    background-size: 40px 40px, 40px 40px; opacity:0; transition: opacity .2s ease; }
  .grid.show{ opacity:1; }

  @media (max-width: 1100px){
    .shell{ grid-template-columns: 1fr; }
    .drawer{ grid-column: 1 / -1; width:100%; }
  }
</style>

<div class="shell expanded">
  <div class="hero">
    <span class="badge"><span class="dot"></span> Interactive Picker</span>
    <div class="hero-title">Precision point selection</div>
    <div class="spacer"></div>
    <div class="hero-sub">Left=POS, Right=NEG, Undo <span class="key">U</span>/<span class="key">⌘Z</span>, Zoom=Wheel, Pan=<span class="key">Space</span>+Drag</div>
  </div>

  <div class="main">
    <div class="head">
      <div class="hero-title" id="legend">Loading…</div>
      <div class="spacer"></div>
      <div class="pill"><span id="count">0</span> points</div>
    </div>
    <div class="stage" id="stage">
      <canvas id="c"></canvas>
      <div class="crosshair" id="xh"><div class="h"></div><div class="v"></div></div>
      <div class="grid" id="grid"></div>

      <div class="overlay-ui">
        <div class="seg-switch" role="group" aria-label="POS/NEG mode">
          <button id="posBtn" class="active pos" type="button">POS</button>
          <button id="negBtn" class="neg" type="button">NEG</button>
        </div>
        <button class="btn-ghost btn" id="undoBtn" onclick="undo()">Undo (U)</button>
        <button class="btn-ghost btn" onclick="toggleGrid()">Grid</button>
        <button class="btn-ghost btn" onclick="clearAll()">Clear</button>
        <button class="btn-primary btn" onclick="save()">Save</button>
      </div>

      <div class="coords" id="coords">x: –, y: – | zoom: 1.00×</div>
    </div>
  </div>

  <div class="drawer">
    <div class="panel">
      <div class="hd"><strong>Points</strong><div class="spacer"></div><span class="badge-mini" id="miniCount">0</span></div>
      <div class="bd" id="list"></div>
    </div>
  </div>
</div>

<div id="toast" class="toast">Saved</div>

<div id="previewModal" style="
  position:fixed; inset:0; display:none; align-items:center; justify-content:center;
  background:rgba(0,0,0,0.55); backdrop-filter:blur(10px); z-index:1000;">
  <div style="max-width:1100px; width:90%; max-height:80vh; overflow:auto;
              background:rgba(15,15,20,0.95); border-radius:18px;
              border:1px solid rgba(255,255,255,0.12); padding:16px 18px; color:#f5f5f7;">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
      <div style="font-weight:700; font-size:16px;">Preview masks</div>
      <div style="flex:1 1 auto;"></div>
      <span style="color:#a3a3aa; font-size:13px;">Check if the masks look okay before continuing.</span>
    </div>
    <div id="previewGrid" style="
            display:flex; flex-direction:column; gap:10px;
            margin-bottom:12px;">
    </div>
    <div style="display:flex; justify-content:flex-end; gap:10px;">
      <button id="restartBtn" class="btn-ghost btn" type="button">Start over</button>
      <button id="confirmBtn" class="btn-primary btn" type="button">Looks good, continue</button>
    </div>
  </div>
</div>

<div id="loadingOverlay" style="
  position:fixed; inset:0;
  display:none;
  align-items:center; justify-content:center;
  background:rgba(0,0,0,0.55);
  backdrop-filter:blur(10px);
  z-index:900;">
  <div style="
      display:flex; flex-direction:column; align-items:center; gap:10px;
      padding:16px 18px;
      border-radius:14px;
      border:1px solid rgba(255,255,255,0.2);
      background:rgba(15,15,20,0.96);
      color:#f5f5f7;">
    
    <!-- Bouncing dots row -->
    <div style="display:flex; gap:6px; align-items:center; justify-content:center;">
      <div class="dot-bounce" style="
          width:8px; height:8px; border-radius:50%;
          background:#0a84ff; animation-delay:0s;"></div>
      <div class="dot-bounce" style="
          width:8px; height:8px; border-radius:50%;
          background:#0a84ff; animation-delay:0.12s;"></div>
      <div class="dot-bounce" style="
          width:8px; height:8px; border-radius:50%;
          background:#0a84ff; animation-delay:0.24s;"></div>
    </div>

    <div id="loadingLabel" style="font-weight:600; font-size:14px;">
      Generating preview…
    </div>
  </div>
</div>

<div id="doneOverlay" style="
  position:fixed; inset:0;
  display:none;
  align-items:center; justify-content:center;
  background:rgba(0,0,0,0.55);
  backdrop-filter:blur(10px);
  z-index:1100;">
  <div style="
      max-width:460px; width:90%;
      padding:18px 20px;
      border-radius:18px;
      border:1px solid rgba(255,255,255,0.18);
      background:rgba(15,15,20,0.97);
      color:#f5f5f7;
      box-shadow:0 30px 80px rgba(0,0,0,0.55);">

    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
      <div style="
          width:24px; height:24px; border-radius:999px;
          display:flex; align-items:center; justify-content:center;
          background:rgba(52,199,89,0.16);">
        <div style="
            width:10px; height:10px; border-radius:50%;
            background:#34c759; box-shadow:0 0 0 5px rgba(52,199,89,0.32);">
        </div>
      </div>
      <div style="font-weight:700; font-size:16px;">All set</div>
    </div>

    <div style="font-size:14px; color:#d1d1d6; margin-bottom:14px;">
      Your prompts have been saved and the pipeline can continue.
      You can safely close this tab and go back to the terminal.
    </div>

    <div style="display:flex; justify-content:flex-end;">
      <button type="button"
              onclick="document.getElementById('doneOverlay').style.display='none';"
              class="btn-ghost btn"
              style="min-width:110px; text-align:center;">
        Close
      </button>
    </div>
  </div>
</div>



<script>
/* --------------------------- State & DOM refs --------------------------- */
const frames   = {{ frames|tojson }};
const legend   = document.getElementById('legend');
const countEl  = document.getElementById('count');
const miniCount= document.getElementById('miniCount');
const listEl   = document.getElementById('list');
const toast    = document.getElementById('toast');
const coords   = document.getElementById('coords');
const grid     = document.getElementById('grid');
const c        = document.getElementById('c');
const ctx      = c.getContext('2d');
const stage    = document.getElementById('stage');
const crosshair= document.getElementById('xh');

const imgEl = new Image();
let points = {};       // points[0] = [{x,y,l}, ...]
let mode = 1;          // 1=POS, 0=NEG
let scale = 1, panX = 0, panY = 0;
let isPanning = false, panStartX = 0, panStartY = 0, worldStartX = 0, worldStartY = 0;
let spaceDown = false;

/* small ripple animation on click */
const ripples = [];
function addRipple(wx, wy, color){ ripples.push({x:wx, y:wy, r:0, color, alpha:0.35}); }
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingLabel   = document.getElementById('loadingLabel');

function setLoading(on, msg = 'Generating preview…') {
  if (loadingOverlay) {
    loadingOverlay.style.display = on ? 'flex' : 'none';
  }
  if (loadingLabel && msg) {
    loadingLabel.textContent = msg;
  }
}

/* ----------------------------- Init & Fit ------------------------------ */
function loadFrame0(){
  if (!frames.length){ legend.textContent = "No frames found"; return; }
  const f = frames[0];
  imgEl.onload = () => {
    // canvas world size = image intrinsic size
    c.width = imgEl.naturalWidth; c.height = imgEl.naturalHeight;
    fitToStage(); redraw();
    legend.textContent = "Annotating " + f.split('/').pop();
  };
  imgEl.src = "/frame?i=0";
}

function fitToStage(){
  // Compute a scale that fits canvas inside the visible stage, centered
  const box = stage.getBoundingClientRect();
  const pad = 8;
  const availW = Math.max(200, box.width  - pad*2);
  const availH = Math.max(200, box.height - pad*2);
  const sx = availW / c.width;
  const sy = availH / c.height;
  scale = Math.max(0.12, Math.min(sx, sy));
  panX = (box.width  - c.width  * scale) / 2;
  panY = (box.height - c.height * scale) / 2;
}

/* ------------------------------- Draw ---------------------------------- */
function updateCount(){
  const n = (points[0] || []).length;
  countEl.textContent = n; miniCount.textContent = n;
}
function renderList(){
  const arr = points[0] || [];
  listEl.innerHTML = arr.map((p,i)=>`
    <div class="row"><div>#${i+1} · ${p.l? 'POS':'NEG'}</div>
    <div class="badge-mini">x:${p.x}, y:${p.y}</div></div>`).join('');
}
function redraw(){
  if (!imgEl.complete) return;
  ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,c.width,c.height);
  ctx.setTransform(scale,0,0,scale,panX,panY);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(imgEl, 0, 0);

  const arr = points[0] || [];
  for (const p of arr){
    ctx.beginPath(); ctx.arc(p.x, p.y, 7/scale, 0, 2*Math.PI);
    ctx.lineWidth = 2/scale; ctx.strokeStyle = p.l ? "#34c759" : "#ff3b30"; ctx.stroke();
    ctx.beginPath(); ctx.arc(p.x, p.y, 2.6/scale, 0, 2*Math.PI); ctx.fillStyle = p.l ? "#34c759" : "#ff3b30"; ctx.fill();
  }
  for (const r of ripples){
    ctx.beginPath(); ctx.arc(r.x, r.y, (r.r)/scale, 0, 2*Math.PI);
    ctx.lineWidth = 2/scale; ctx.strokeStyle = r.color.replace('1)', r.alpha + ')'); ctx.stroke();
    r.r += 12; r.alpha *= 0.92;
  }
  for (let i = ripples.length-1; i>=0; i--){ if (ripples[i].alpha < 0.04) ripples.splice(i,1); }

  updateCount(); renderList();
}

/* ------------------------- Coordinate helpers -------------------------- */
function getCanvasScreenXY(e){ const r = c.getBoundingClientRect(); return { sx: e.clientX - r.left, sy: e.clientY - r.top }; }
function screenToWorldCanvas(sx, sy){ return { x: Math.round((sx - panX)/scale), y: Math.round((sy - panY)/scale) }; }

/* ------------------------------- Events -------------------------------- */
new ResizeObserver(()=>{ fitToStage(); redraw(); }).observe(stage);

// Crosshair + live coords (measured on canvas)
c.addEventListener('mousemove', (e)=>{
  const { sx, sy } = getCanvasScreenXY(e);
  const w = screenToWorldCanvas(sx, sy);
  const r = c.getBoundingClientRect(), sr = stage.getBoundingClientRect();
  crosshair.style.setProperty('--x', `${(r.left - sr.left) + sx}px`);
  crosshair.style.setProperty('--y', `${(r.top  - sr.top ) + sy}px`);
  coords.textContent = `x: ${Math.max(0,Math.min(c.width, w.x))}, y: ${Math.max(0,Math.min(c.height, w.y))} | zoom: ${scale.toFixed(2)}×`;
});

// Wheel zoom anchored at cursor
c.addEventListener('wheel', (e)=>{
  if (!imgEl.complete) return;
  e.preventDefault();
  const { sx, sy } = getCanvasScreenXY(e);
  const before = screenToWorldCanvas(sx, sy);
  const newScale = Math.min(12, Math.max(0.1, scale * (1 - Math.sign(e.deltaY)*0.12)));
  scale = newScale;
  panX = sx - (before.x * scale);
  panY = sy - (before.y * scale);
  redraw();
},{passive:false});

// Space = pan mode
window.addEventListener('keydown', (e)=>{ if (e.code === 'Space') spaceDown = true; });
window.addEventListener('keyup',   (e)=>{ if (e.code === 'Space') spaceDown = false; });

// Mouse down (pan or add point)
c.addEventListener('mousedown', (e)=>{
  const { sx, sy } = getCanvasScreenXY(e);
  if (spaceDown){
    isPanning = true; panStartX = sx; panStartY = sy; worldStartX = panX; worldStartY = panY; stage.style.cursor = 'grabbing'; return;
  }
  if (!points[0]) points[0] = [];
  const w = screenToWorldCanvas(sx, sy);
  const label = (e.button === 2) ? 0 : mode;  // right-click forces NEG
  points[0].push({x: w.x, y: w.y, l: label});
  addRipple(w.x, w.y, label ? 'rgba(52,199,89,1)' : 'rgba(255,59,48,1)');
  redraw();
});
c.addEventListener('contextmenu', e => e.preventDefault());

// Pan drag
c.addEventListener('mousemove', (e)=>{
  if (!isPanning) return;
  const { sx, sy } = getCanvasScreenXY(e);
  panX = worldStartX + (sx - panStartX);
  panY = worldStartY + (sy - panStartY);
  redraw();
});
window.addEventListener('mouseup', ()=>{ isPanning = false; stage.style.cursor = 'crosshair'; });

// Undo
function undo(){ if (!points[0] || !points[0].length) return; points[0].pop(); redraw(); }
window.addEventListener('keydown', (e)=>{
  const tgt = (e.target.tagName||'').toLowerCase();
  if (tgt === 'input' || tgt === 'textarea' || e.isComposing) return;
  const isUndo = (e.key === 'u' || e.key === 'U') || ((e.ctrlKey||e.metaKey) && e.key.toLowerCase()==='z');
  if (isUndo){ e.preventDefault(); undo(); }
});

// Mode toggle
const posBtn = document.getElementById('posBtn');
const negBtn = document.getElementById('negBtn');
posBtn.addEventListener('click', ()=>{ mode=1; posBtn.classList.add('active','pos'); negBtn.classList.remove('active'); });
negBtn.addEventListener('click', ()=>{ mode=0; negBtn.classList.add('active','neg'); posBtn.classList.remove('active'); });

// Grid & clear
function toggleGrid(){ grid.classList.toggle('show'); }
function clearAll(){ if (confirm('Clear all points?')){ points[0] = []; redraw(); } }

// Save
function showToast(msg="Saved"){ toast.textContent = msg; toast.classList.add('show'); setTimeout(()=>toast.classList.remove('show'), 1300); }
const previewModal = document.getElementById('previewModal');
const previewGrid  = document.getElementById('previewGrid');
const doneOverlay  = document.getElementById('doneOverlay');


async function save(){
  const payload = { points };
  showToast('Generating preview…');
  setLoading(true, 'Generating preview…');

  try{
    const r = await fetch('/save', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });

    if (!r.ok){
      showToast('Save failed');
      alert('Save failed: ' + await r.text());
      return;
    }

    const data = await r.json();
    if (!data.ok){
      showToast('Save failed');
      alert('Save failed: ' + (data.error || 'Unknown error'));
      return;
    }

    // data.previews = ["/preview/....png", ...]
    const previews = data.previews || [];
        if (!previews.length){
          showToast('No preview generated');
          alert('No preview images generated. Check the backend preview logic.');
          return;
        }


        const main = previews[0];
        const thumbs = previews.slice(1);  // random 5

        let html = "";

    
        html += `
          <div style="border-radius:14px; overflow:hidden;
                      border:1px solid rgba(255,255,255,0.18);
                      background:#000;">
            <img src="${main}"
                style="display:block; width:100%; max-height:420px;
                        object-fit:contain;">
          </div>
        `;

        
        if (thumbs.length){
          html += `
            <div style="display:flex; flex-wrap:wrap; gap:10px;">
              ${thumbs.map(url => `
                <div style="flex:1 1 160px; max-width:220px;
                            border-radius:12px; overflow:hidden;
                            border:1px solid rgba(255,255,255,0.12);
                            background:#000;">
                  <img src="${url}" style="display:block; width:100%; height:auto;">
                </div>
              `).join('')}
            </div>
          `;
        }

    previewGrid.innerHTML = html;


    // Show modal
    previewModal.style.display = 'flex';
    setLoading(false);
    showToast('Preview ready');


  }catch(err){
    setLoading(false);
    showToast('Save failed');
    alert('Save failed: ' + err);
  }
}
document.getElementById('confirmBtn').addEventListener('click', async () => {
  try{
    const r = await fetch('/confirm', { method:'POST' });
    const data = await r.json();
    if (!r.ok || !data.ok){
      alert('Failed to confirm: ' + (data.error || 'unknown'));
      return;
    }
   
    previewModal.style.display = 'none';
    showToast('Confirmed');
    doneOverlay.style.display = 'flex';
  }catch(err){
    alert('Failed to confirm: ' + err);
  }
});

document.getElementById('restartBtn').addEventListener('click', async () => {
  if (!confirm('Discard these prompts and start over?')) return;
  try{
    const r = await fetch('/restart', { method:'POST' });
    const data = await r.json();
    if (!r.ok || !data.ok){
      alert('Failed to restart: ' + (data.error || 'unknown'));
      return;
    }
    showToast('Restarted');
    previewModal.style.display = 'none';
    // Clear all points locally too
    points[0] = [];
    redraw();
  }catch(err){
    alert('Failed to restart: ' + err);
  }
});



// Start
function tick(){ requestAnimationFrame(tick); if (ripples.length) redraw(); }
tick(); loadFrame0();

</script>
"""
DONE_HTML = """
  <!doctype html><meta charset="utf-8">
  <title>Point Picker — {{ds}}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <style>
    :root{
      --bg: #0b0b0d; --ink:#f5f5f7; --muted:#a3a3aa;
      --card: rgba(22,22,24,0.66); --border: rgba(255,255,255,0.08);
      --accent:#0a84ff; --shadow: 0 30px 80px rgba(0,0,0,0.45); --glass: saturate(180%) blur(22px);
    }
    @media (prefers-color-scheme: light){
      :root{ --bg:#f5f5f7; --ink:#0b0b0d; --muted:#6e6e73; --card:rgba(255,255,255,0.72); --border:rgba(0,0,0,0.08); --shadow:0 30px 70px rgba(0,0,0,0.12); }
    }
    *{ box-sizing:border-box }
    html,body{ height:100%; }
    body{
      margin:0; color:var(--ink);
      background:
        radial-gradient(1200px 800px at 10% -10%, rgba(10,132,255,0.14), transparent 60%),
        radial-gradient(1000px 700px at 120% 110%, rgba(94,92,230,0.14), transparent 60%),
        var(--bg);
      font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display","Segoe UI",Roboto,Helvetica,Arial;
      display:flex; align-items:center; justify-content:center; padding:24px;
    }
    .card{
      max-width:560px; width:100%;
      border-radius:18px; border:1px solid var(--border);
      background:var(--card); backdrop-filter:var(--glass); box-shadow:var(--shadow);
      padding:18px 20px; display:flex; flex-direction:column; gap:10px;
    }
    .title{ font-weight:800; font-size:18px; letter-spacing:.2px; }
    .muted{ color:var(--muted); }
    .badge{ display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border:1px solid var(--border);
            border-radius:999px; background:rgba(255,255,255,0.06); color:var(--muted); font-weight:600; }
    .dot-ok{
      width:10px; height:10px; border-radius:50%;
      background:#34c759; box-shadow:0 0 0 5px rgba(52,199,89,0.25);
    }
    .files{ margin-top:6px; font-size:13px; color:var(--muted); }
    .files div{ margin:2px 0; }
  </style>

  <div class="card">
    <div class="badge">
      <span class="dot-ok"></span> Using existing prompts
    </div>
    <div class="title">You’re good to go ✅</div>
    <div class="muted">
      The existing <code>prompts.json</code> for <strong>{{ds}}</strong> will be used.
      You can now close this tab and return to the terminal.
    </div>

    <div class="files">
      <div><strong>PROMPTS_JSON:</strong> {{prompts}}</div>
      <div><strong>DONE_FLAG:</strong> {{done}}</div>
      <div><strong>USE_EXISTING:</strong> {{use_existing}}</div>
    </div>
  </div>
  """

# =============================================================================
# App & Utilities
# =============================================================================

app = Flask(__name__)


def gather_frames(dir_path: Path | str) -> List[str]:
    """
    Collect image files from a directory (jpg/jpeg/png, case-insensitive).
    Returns absolute paths sorted lexicographically.
    """
    p = Path(dir_path)
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files: List[str] = []
    for pat in patterns:
        files.extend([str(f) for f in p.glob(pat)])
    return sorted(files)


def resolve_frames() -> List[str]:
    """
    Try INPUT first (e.g. /data/in/dress).
    If empty, also try <INPUT>_indexed (e.g. /data/in/dress_indexed).
    """
    frames = gather_frames(INPUT)
    if not frames:
        maybe_indexed = f"{str(INPUT).rstrip('/')}{INDEX_SUFFIX}"
        frames = gather_frames(maybe_indexed)
    return frames


FRAMES: List[str] = resolve_frames()
def run_preview_masks(num_frames: int = 6) -> List[str]:
    """
    Run a small SAM2 preview:
      - reads prompts.json in OUT_ROOT
      - runs segmentation/propagation on a few frames
      - writes color cutouts into PREVIEW_DIR

    Returns: list of filenames (just the name, e.g. "000000.jpg").
    """

    # Clear previous previews
    for f in PREVIEW_DIR.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass

    cmd = [
        "python3",
        "app/video_predict.py",
        "--preview",
        "--preview-num-frames", str(num_frames),
        "--preview-out", str(PREVIEW_DIR),
    ]

    
    env = os.environ.copy()
    env["QUIET"] = "0"

    try:
        print(f"[preview] running: {' '.join(cmd)}  -> {PREVIEW_DIR}", flush=True)
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[preview] video_predict.py failed: {e}", flush=True)
        return []

    # Collect the files that were written into PREVIEW_DIR
    previews: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        for f in sorted(PREVIEW_DIR.glob(ext)):
            previews.append(f.name)   

    print(f"[preview] found {len(previews)} preview files", flush=True)
    return previews





def _json_ok(**payload: Any):
    return jsonify({"ok": True, **payload})


def _json_err(msg: str, http: int = 400):
    return jsonify({"ok": False, "error": msg}), http


# =============================================================================
# Routes
# =============================================================================

@app.route("/")
def home():
    """
    Home (Apple-like) — shows dataset name, frame count, and actions:
    - Use existing prompts.json (if present)
    - Create new (opens picker)
    """
    return render_template_string(
        HOME_HTML,
        ds=DS_NAME,
        nframes=len(FRAMES),
        exists=PROMPTS_JSON.is_file(),
        prompts=str(PROMPTS_JSON),
        done=str(DONE_FLAG),
        use_existing=str(USE_EXISTING),
    )


@app.post("/use_existing")
def use_existing():
    """
    User chose to reuse existing prompts.json.
    We set USE_EXISTING + DONE_FLAG and show a nice “done” screen.
    """
    USE_EXISTING.touch()
    DONE_FLAG.touch()
    return render_template_string(
        DONE_HTML,
        ds=DS_NAME,
        prompts=str(PROMPTS_JSON),
        done=str(DONE_FLAG),
        use_existing=str(USE_EXISTING),
    )



@app.post("/create_new")
def create_new():
    """
    Remove previous prompts and use_existing flag, then redirect to /pick.
    """
    try:
        PROMPTS_JSON.unlink(missing_ok=True)
        USE_EXISTING.unlink(missing_ok=True)
    except Exception as e:
        return _json_err(str(e), http=500)
    return redirect("/pick")


@app.get("/pick")
def pick():
    """
    Interactive picker view (Apple-like).
    Loads only the list of frames to show the first image.
    """
    names = [Path(p).name for p in FRAMES]
    return render_template_string(
        PICK_HTML,
        frames=FRAMES,
        idx=1,
        total=len(FRAMES),
        name=names[0] if names else "",
    )


@app.get("/preview/<path:name>")
def preview_image(name: str):
    """
    Serve preview masked images from PREVIEW_DIR.
    """
    fp = PREVIEW_DIR / name
    if not fp.is_file():
        return _json_err("Preview image not found", http=404)
    return send_from_directory(PREVIEW_DIR, fp.name)



@app.post("/save")
def save():
    """
    Persist prompts.json in the indexed output directory AND
    run a small SAM2 preview on a few frames.

    Important:
      - We DO NOT create DONE_FLAG here anymore.
      - DONE_FLAG will be created only in /confirm when the user accepts the preview.
    """
    if not FRAMES:
        return _json_err("No frames found", http=400)

    try:
        data = request.get_json(force=True, silent=True) or {}
        points_dict: Dict[str, List[Dict[str, int]]] = data.get("points", {})
        frame_idx = 0

        # Clients may use numeric or string keys ("0")
        raw_arr = points_dict.get(str(frame_idx), points_dict.get(frame_idx, []))
        pts: List[List[float]] = []
        labs: List[int] = []

        for p in raw_arr:
            pts.append([float(p["x"]), float(p["y"])])
            labs.append(int(p["l"]))

        # Read image size for metadata and fallback center
        from PIL import Image
        with Image.open(FRAMES[frame_idx]) as im:
            w, h = im.size

        if not pts:
            # Fallback: if no clicks, place 1 POS at center to avoid empty JSON
            pts = [[w // 2, h // 2]]
            labs = [1]

        PROMPTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with PROMPTS_JSON.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "frame_idx": int(frame_idx),
                    "obj_id": 1,
                    "points": pts,
                    "labels": labs,
                    "image_w": int(w),
                    "image_h": int(h),
                    "source": Path(FRAMES[frame_idx]).name,
                },
                f,
                indent=2,
            )

        # --- NEW: run preview masks on 1+5 frames ---
        preview_files = run_preview_masks(num_frames=6)
        preview_urls = [f"/preview/{name}" for name in preview_files]

        # DO NOT touch DONE_FLAG here; we wait for /confirm
        return _json_ok(path=str(PROMPTS_JSON), previews=preview_urls)

    except Exception as e:
        return _json_err(str(e), http=500)

@app.post("/confirm")
def confirm():
    """
    User accepted the preview: signal the runner to continue by
    creating DONE_FLAG.
    """
    try:
        DONE_FLAG.touch()
        return _json_ok(msg="confirmed")
    except Exception as e:
        return _json_err(str(e), http=500)


@app.post("/restart")
def restart():
    """
    User rejected the preview: delete prompts and previews,
    so they can click new points.
    """
    try:
        PROMPTS_JSON.unlink(missing_ok=True)
        for f in PREVIEW_DIR.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        # Also ensure flags are clean
        DONE_FLAG.unlink(missing_ok=True)
        USE_EXISTING.unlink(missing_ok=True)
        return _json_ok(msg="restarted")
    except Exception as e:
        return _json_err(str(e), http=500)

@app.get("/frame")
def frame():
    """
    Serve a source frame image by index (used by the canvas).
    """
    i_str = request.args.get("i", "0")
    try:
        idx = int(i_str)
    except ValueError:
        return _json_err("Invalid frame index", http=400)

    if idx < 0 or idx >= len(FRAMES):
        return _json_err("Frame index out of range", http=404)

    fp = Path(FRAMES[idx])
    return send_from_directory(fp.parent, fp.name)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # NOTE: Keep debug=False for Docker usage; change locally if desired.
    app.run(host="0.0.0.0", port=5000, debug=False)
