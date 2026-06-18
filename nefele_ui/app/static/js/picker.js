(() => {
  'use strict';

  // ---- State & DOM refs ----
  const frames = JSON.parse(document.getElementById('frames-data').textContent);
  const legend = document.getElementById('legend');
  const countEl = document.getElementById('count');
  const miniCount = document.getElementById('miniCount');
  const listEl = document.getElementById('list');
  const toast = document.getElementById('toast');
  const coords = document.getElementById('coords');
  const grid = document.getElementById('grid');
  const c = document.getElementById('c');
  const ctx = c.getContext('2d');
  const stage = document.getElementById('stage');
  const crosshair = document.getElementById('xh');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const loadingLabel = document.getElementById('loadingLabel');
  const previewModal = document.getElementById('previewModal');
  const previewGrid = document.getElementById('previewGrid');
  const doneOverlay = document.getElementById('doneOverlay');

  const imgEl = new Image();
  const points = {};
  let currentFrameIdx = 0;
  let mode = 1;
  const frameIndicator = document.getElementById('frameIndicator');
  const prevFrameBtn = document.getElementById('prevFrameBtn');
  const nextFrameBtn = document.getElementById('nextFrameBtn');
  let scale = 1, panX = 0, panY = 0;
  let isPanning = false, panStartX = 0, panStartY = 0, worldStartX = 0, worldStartY = 0;
  let spaceDown = false;
  const ripples = [];

  // ---- Helpers ----
  function addRipple(wx, wy, color) {
    ripples.push({ x: wx, y: wy, r: 0, color, alpha: 0.35 });
  }

  function setLoading(on, msg = 'Generating preview…') {
    loadingOverlay.classList.toggle('show', !!on);
    if (msg) loadingLabel.textContent = msg;
  }

  function showToast(msg = 'Saved') {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 1300);
  }

  // ---- Init & fit ----
  function updateFrameIndicator() {
    if (frameIndicator) {
      frameIndicator.textContent = `Frame ${currentFrameIdx + 1} / ${frames.length}`;
    }
    if (prevFrameBtn) prevFrameBtn.disabled = currentFrameIdx <= 0;
    if (nextFrameBtn) nextFrameBtn.disabled = currentFrameIdx >= frames.length - 1;
  }

  function loadFrame(idx) {
    if (!frames.length) { legend.textContent = 'No frames found'; return; }
    if (idx < 0 || idx >= frames.length) return;
    currentFrameIdx = idx;
    const f = frames[idx];
    imgEl.onload = () => {
      c.width = imgEl.naturalWidth;
      c.height = imgEl.naturalHeight;
      fitToStage();
      redraw();
      legend.textContent = 'Annotating ' + f.split('/').pop();
      updateFrameIndicator();
    };
    imgEl.src = '/frame?i=' + idx;
  }

  function changeFrame(newIdx) {
    if (newIdx === currentFrameIdx) return;
    if (newIdx < 0 || newIdx >= frames.length) return;
    const existing = points[currentFrameIdx] || [];
    if (existing.length && !confirm(`You have ${existing.length} point(s) on this frame. Switching will discard them. Continue?`)) {
      return;
    }
    points[currentFrameIdx] = [];
    loadFrame(newIdx);
  }

  function fitToStage() {
    const box = stage.getBoundingClientRect();
    const pad = 8;
    const availW = Math.max(200, box.width - pad * 2);
    const availH = Math.max(200, box.height - pad * 2);
    const sx = availW / c.width;
    const sy = availH / c.height;
    scale = Math.max(0.12, Math.min(sx, sy));
    panX = (box.width - c.width * scale) / 2;
    panY = (box.height - c.height * scale) / 2;
  }

  // ---- Draw ----
  function updateCount() {
    const n = (points[currentFrameIdx] || []).length;
    countEl.textContent = n;
    miniCount.textContent = n;
  }

  function renderList() {
    const arr = points[currentFrameIdx] || [];
    listEl.innerHTML = arr.map((p, i) => `
      <div class="row"><div>#${i + 1} · ${p.l ? 'POS' : 'NEG'}</div>
      <div class="badge-mini">x:${p.x}, y:${p.y}</div></div>`).join('');
  }

  function redraw() {
    if (!imgEl.complete) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.setTransform(scale, 0, 0, scale, panX, panY);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(imgEl, 0, 0);

    const arr = points[currentFrameIdx] || [];
    for (const p of arr) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7 / scale, 0, 2 * Math.PI);
      ctx.lineWidth = 2 / scale;
      ctx.strokeStyle = p.l ? '#34c759' : '#ff3b30';
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.6 / scale, 0, 2 * Math.PI);
      ctx.fillStyle = p.l ? '#34c759' : '#ff3b30';
      ctx.fill();
    }
    for (const r of ripples) {
      ctx.beginPath();
      ctx.arc(r.x, r.y, r.r / scale, 0, 2 * Math.PI);
      ctx.lineWidth = 2 / scale;
      ctx.strokeStyle = r.color.replace('1)', r.alpha + ')');
      ctx.stroke();
      r.r += 12;
      r.alpha *= 0.92;
    }
    for (let i = ripples.length - 1; i >= 0; i--) {
      if (ripples[i].alpha < 0.04) ripples.splice(i, 1);
    }

    updateCount();
    renderList();
  }

  // ---- Coordinate helpers ----
  function getCanvasScreenXY(e) {
    const r = c.getBoundingClientRect();
    return { sx: e.clientX - r.left, sy: e.clientY - r.top };
  }

  function screenToWorldCanvas(sx, sy) {
    return {
      x: Math.round((sx - panX) / scale),
      y: Math.round((sy - panY) / scale),
    };
  }

  // ---- Events ----
  new ResizeObserver(() => { fitToStage(); redraw(); }).observe(stage);

  c.addEventListener('mousemove', (e) => {
    const { sx, sy } = getCanvasScreenXY(e);
    const w = screenToWorldCanvas(sx, sy);
    const r = c.getBoundingClientRect();
    const sr = stage.getBoundingClientRect();
    crosshair.style.setProperty('--x', `${(r.left - sr.left) + sx}px`);
    crosshair.style.setProperty('--y', `${(r.top - sr.top) + sy}px`);
    coords.textContent =
      `x: ${Math.max(0, Math.min(c.width, w.x))}, y: ${Math.max(0, Math.min(c.height, w.y))} | zoom: ${scale.toFixed(2)}×`;
  });

  c.addEventListener('wheel', (e) => {
    if (!imgEl.complete) return;
    e.preventDefault();
    const { sx, sy } = getCanvasScreenXY(e);
    const before = screenToWorldCanvas(sx, sy);
    const newScale = Math.min(12, Math.max(0.1, scale * (1 - Math.sign(e.deltaY) * 0.12)));
    scale = newScale;
    panX = sx - (before.x * scale);
    panY = sy - (before.y * scale);
    redraw();
  }, { passive: false });

  window.addEventListener('keydown', (e) => { if (e.code === 'Space') spaceDown = true; });
  window.addEventListener('keyup', (e) => { if (e.code === 'Space') spaceDown = false; });

  c.addEventListener('mousedown', (e) => {
    const { sx, sy } = getCanvasScreenXY(e);
    if (spaceDown) {
      isPanning = true;
      panStartX = sx; panStartY = sy;
      worldStartX = panX; worldStartY = panY;
      stage.style.cursor = 'grabbing';
      return;
    }
    if (!points[currentFrameIdx]) points[currentFrameIdx] = [];
    const w = screenToWorldCanvas(sx, sy);
    const label = (e.button === 2) ? 0 : mode;
    points[currentFrameIdx].push({ x: w.x, y: w.y, l: label });
    addRipple(w.x, w.y, label ? 'rgba(52,199,89,1)' : 'rgba(255,59,48,1)');
    redraw();
  });

  c.addEventListener('contextmenu', (e) => e.preventDefault());

  c.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    const { sx, sy } = getCanvasScreenXY(e);
    panX = worldStartX + (sx - panStartX);
    panY = worldStartY + (sy - panStartY);
    redraw();
  });

  window.addEventListener('mouseup', () => { isPanning = false; stage.style.cursor = 'crosshair'; });

  function undo() {
    const arr = points[currentFrameIdx];
    if (!arr || !arr.length) return;
    arr.pop();
    redraw();
  }

  window.addEventListener('keydown', (e) => {
    const tgt = (e.target.tagName || '').toLowerCase();
    if (tgt === 'input' || tgt === 'textarea' || e.isComposing) return;
    const isUndo = (e.key === 'u' || e.key === 'U') ||
                   ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z');
    if (isUndo) { e.preventDefault(); undo(); }
  });

  // ---- Toolbar wiring ----
  const posBtn = document.getElementById('posBtn');
  const negBtn = document.getElementById('negBtn');
  posBtn.addEventListener('click', () => {
    mode = 1;
    posBtn.classList.add('active', 'pos');
    negBtn.classList.remove('active');
  });
  negBtn.addEventListener('click', () => {
    mode = 0;
    negBtn.classList.add('active', 'neg');
    posBtn.classList.remove('active');
  });

  document.getElementById('undoBtn').addEventListener('click', undo);
  document.getElementById('gridBtn').addEventListener('click', () => grid.classList.toggle('show'));
  document.getElementById('clearBtn').addEventListener('click', () => {
    if (confirm('Clear all points?')) { points[currentFrameIdx] = []; redraw(); }
  });

  // ---- Save / confirm / restart ----
  function renderPreviews(previews) {
    const main = previews[0];
    const thumbs = previews.slice(1);

    let html = `
      <div class="preview-main">
        <img src="${main}">
      </div>`;
    if (thumbs.length) {
      html += `<div class="preview-thumbs">
        ${thumbs.map(url => `<div class="preview-thumb"><img src="${url}"></div>`).join('')}
      </div>`;
    }
    previewGrid.innerHTML = html;
  }

  // Poll /save/status until preview_ready, then show the preview modal.
  async function pollPreview(jobId, attempts = 0) {
    const MAX_ATTEMPTS = 180; // 180 × 3s = 9 min ceiling
    if (attempts >= MAX_ATTEMPTS) {
      setLoading(false);
      showToast('Timed out');
      alert('Preview timed out — the SAM2 worker may not be running.');
      return;
    }
    try {
      const r = await fetch(`/save/status?job_id=${encodeURIComponent(jobId)}`, { cache: 'no-store' });
      const data = await r.json().catch(() => null);
      if (!r.ok || !data || !data.ok) {
        setLoading(false);
        showToast('Preview failed');
        alert('Preview error: ' + ((data && data.error) || `HTTP ${r.status}`));
        return;
      }
      if (data.pending) {
        // Still waiting — update message and retry
        const dots = '.'.repeat((attempts % 3) + 1);
        setLoading(true, `Generating Previews${dots} (${attempts * 3}s)`);
        setTimeout(() => pollPreview(jobId, attempts + 1), 3000);
        return;
      }
      // Preview ready
      setLoading(false);
      const previews = data.previews || [];
      if (!previews.length) {
        showToast('No preview images');
        alert('Worker finished but returned no preview images.');
        return;
      }
      renderPreviews(previews);
      previewModal.classList.add('show');
      showToast('Preview ready');
    } catch (err) {
      setLoading(false);
      showToast('Network error');
      alert('Polling failed: ' + err);
    }
  }

  async function save() {
    const payload = { points, frame_idx: currentFrameIdx };
    showToast('Submitting…');
    setLoading(true, 'Submitting points…');
    try {
      const r = await fetch('/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      let data = null;
      try { data = await r.json(); } catch (_) {}

      if (!r.ok || !data || !data.ok) {
        setLoading(false);
        showToast('Save failed');
        alert('Save failed: ' + ((data && data.error) || `HTTP ${r.status}`));
        return;
      }

      // Existing preview resumed — show immediately without polling
      if (!data.pending) {
        setLoading(false);
        const previews = data.previews || [];
        if (!previews.length) {
          showToast('No preview generated');
          return;
        }
        renderPreviews(previews);
        previewModal.classList.add('show');
        showToast('Resumed pending preview');
        if (data.message) alert(data.message);
        return;
      }

      // Job created — poll for preview_ready
      showToast('Job submitted — waiting for worker…');
      setLoading(true, 'Generating Previews…');
      pollPreview(data.job_id);
    } catch (err) {
      setLoading(false);
      showToast('Save failed');
      alert('Save failed: ' + err);
    }
  }

  document.getElementById('saveBtn').addEventListener('click', save);

  document.getElementById('confirmBtn').addEventListener('click', async () => {
    try {
      const r = await fetch('/confirm', { method: 'POST' });
      const data = await r.json();
      if (!r.ok || !data.ok) {
        alert('Failed to confirm: ' + (data.error || 'unknown'));
        return;
      }
      previewModal.classList.remove('show');
      showToast('Confirmed');
      doneOverlay.classList.add('show');
    } catch (err) {
      alert('Failed to confirm: ' + err);
    }
  });

  document.getElementById('restartBtn').addEventListener('click', async () => {
    if (!confirm('Discard these prompts and start over?')) return;
    try {
      const r = await fetch('/restart', { method: 'POST' });
      const data = await r.json();
      if (!r.ok || !data.ok) {
        alert('Failed to restart: ' + (data.error || 'unknown'));
        return;
      }
      showToast('Restarted');
      previewModal.classList.remove('show');
      points[currentFrameIdx] = [];
      redraw();
    } catch (err) {
      alert('Failed to restart: ' + err);
    }
  });

  document.getElementById('closeDoneBtn').addEventListener('click', () => {
    doneOverlay.classList.remove('show');
  });

  // ---- Animation loop ----
  function tick() {
    requestAnimationFrame(tick);
    if (ripples.length) redraw();
  }

  tick();
  if (prevFrameBtn) prevFrameBtn.addEventListener('click', () => changeFrame(currentFrameIdx - 1));
  if (nextFrameBtn) nextFrameBtn.addEventListener('click', () => changeFrame(currentFrameIdx + 1));
  window.addEventListener('keydown', (e) => {
    const tgt = (e.target.tagName || '').toLowerCase();
    if (tgt === 'input' || tgt === 'textarea' || e.isComposing) return;
    if (e.key === 'ArrowLeft')  { e.preventDefault(); changeFrame(currentFrameIdx - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); changeFrame(currentFrameIdx + 1); }
  });

  loadFrame(0);
})();
