(() => {
  'use strict';

  const POLL_INTERVAL_MS = 5000;

  const statusPill = document.getElementById('statusPill');
  const emptyCard = document.getElementById('emptyCard');
  const filesCard = document.getElementById('filesCard');
  const filesList = document.getElementById('filesList');
  const filesCount = document.getElementById('filesCount');
  const stageEls = Array.from(document.querySelectorAll('.stage'));
  const stageMessage = document.getElementById('stageMessage');

  let pollHandle = null;

  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  function formatTime(unixSec) {
    const d = new Date(unixSec * 1000);
    return d.toLocaleString();
  }

  function setStatus(label, cls) {
    statusPill.textContent = label;
    statusPill.classList.remove('ready', 'waiting', 'error');
    if (cls) statusPill.classList.add(cls);
  }

  // Render exactly one card at a time. Errors collapse both so we don't
  // leave a contradictory "pipeline still running" + "0 files ready" mix.
  function setState(name) {
    emptyCard.hidden = name !== 'empty';
    filesCard.hidden = name !== 'files';
  }

  // Update the SAM2 / COLMAP / SuGaR pills based on the pipeline status.
  function applyPipelineStatus(s) {
    if (!s) return;
    const current = typeof s.current === 'number' ? s.current : -1;
    const isError = s.status === 'error';
    const isDone = s.status === 'done';

    stageEls.forEach((el, idx) => {
      el.classList.remove('pending', 'running', 'done', 'error');
      if (isError && idx === current) {
        el.classList.add('error');
      } else if (isDone || idx < current) {
        el.classList.add('done');
      } else if (idx === current && s.status === 'running') {
        el.classList.add('running');
      } else {
        el.classList.add('pending');
      }
    });

    stageMessage.textContent = s.error || s.message || '';
  }

  function renderFiles(files) {
    filesList.innerHTML = '';
    for (const f of files) {
      const li = document.createElement('li');
      li.innerHTML = `
        <span class="kind-pill ${f.kind}"></span>
        <span class="file-name"></span>
        <span class="file-meta">
          <span class="size"></span>
          <span class="dot-sep">·</span>
          <span class="modified"></span>
        </span>
        <a class="results-download" download>Download</a>`;
      li.querySelector('.kind-pill').textContent = f.kind;
      li.querySelector('.file-name').textContent = f.name;
      li.querySelector('.size').textContent = formatBytes(f.size);
      li.querySelector('.modified').textContent = formatTime(f.modified);
      const a = li.querySelector('.results-download');
      a.href = `/results/file/${encodeURI(f.relative)}`;
      filesList.appendChild(li);
    }
    filesCount.textContent = files.length;
  }

  async function fetchPipelineStatus() {
    try {
      const r = await fetch('/pipeline/status', { cache: 'no-store' });
      if (!r.ok) return null;
      const d = await r.json();
      return d.ok ? d : null;
    } catch (_) {
      return null;
    }
  }

  async function tick() {
    try {
      const [filesResp, pipeline] = await Promise.all([
        fetch('/results/files', { cache: 'no-store' }),
        fetchPipelineStatus(),
      ]);
      applyPipelineStatus(pipeline);

      const data = await filesResp.json().catch(() => null);
      if (!filesResp.ok || !data || !data.ok) {
        const msg = (data && data.error) || `HTTP ${filesResp.status}`;
        setStatus(msg, 'error');
        setState('none');
        return;
      }
      if (data.ready) {
        setStatus('Ready', 'ready');
        renderFiles(data.files);
        setState('files');
        // Stop polling once the mesh is on disk AND the pipeline reports done.
        if (pipeline && pipeline.status === 'done' && pollHandle) {
          clearInterval(pollHandle); pollHandle = null;
        }
      } else {
        setStatus('Waiting', 'waiting');
        setState('empty');
      }
    } catch (err) {
      setStatus('Network error', 'error');
      setState('none');
    }
  }

  tick();
  pollHandle = setInterval(tick, POLL_INTERVAL_MS);
})();
