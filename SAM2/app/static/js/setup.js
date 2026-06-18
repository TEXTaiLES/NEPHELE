(() => {
  'use strict';

  // ── Local upload ──────────────────────────────────────────────────────────
  const form          = document.getElementById('setupForm');
  const nameInput     = document.getElementById('datasetName');
  const dropzone      = document.getElementById('dropzone');
  const fileInput     = document.getElementById('fileInput');
  const fileListEl    = document.getElementById('fileList');
  const submitBtn     = document.getElementById('submitBtn');
  const statusLine    = document.getElementById('statusLine');
  const loadingOverlay = document.getElementById('loadingOverlay');

  function selectedModel() {
    const r = document.querySelector('input[name="modelChoice"]:checked');
    return r ? r.value : 'sugar';
  }

  const staged = new DataTransfer();

  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  function refreshList() {
    fileListEl.innerHTML = '';
    const arr = Array.from(staged.files);
    if (arr.length === 0) {
      fileListEl.hidden = true;
      statusLine.textContent = 'No files staged.';
      submitBtn.disabled = !nameInput.value.trim();
      return;
    }
    fileListEl.hidden = false;
    arr.forEach((file, idx) => {
      const li = document.createElement('li');
      li.innerHTML = `
        <span class="name"></span>
        <span class="size"></span>
        <button type="button" class="remove" aria-label="Remove">×</button>`;
      li.querySelector('.name').textContent = file.name;
      li.querySelector('.size').textContent = formatBytes(file.size);
      li.querySelector('.remove').addEventListener('click', () => {
        const next = new DataTransfer();
        Array.from(staged.files).forEach((f, i) => { if (i !== idx) next.items.add(f); });
        staged.items.clear();
        Array.from(next.files).forEach(f => staged.items.add(f));
        fileInput.files = staged.files;
        refreshList();
      });
      fileListEl.appendChild(li);
    });
    statusLine.textContent = `${arr.length} file${arr.length === 1 ? '' : 's'} staged.`;
    submitBtn.disabled = !nameInput.value.trim();
  }

  function addFiles(list) {
    for (const f of list) {
      const isImageMime = f.type && f.type.startsWith('image/');
      const isImageExt  = /\.(jpe?g|png|webp|bmp|tiff?|gif|heic|heif|avif)$/i.test(f.name);
      if (!isImageMime && !isImageExt) continue;
      staged.items.add(f);
    }
    fileInput.files = staged.files;
    refreshList();
  }

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', () => addFiles(fileInput.files));
  ['dragenter', 'dragover'].forEach(ev =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach(ev =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove('drag'); }));
  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });
  nameInput.addEventListener('input', refreshList);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = nameInput.value.trim();
    if (!name) return;
    const fd = new FormData();
    fd.append('name', name);
    fd.append('model', selectedModel());
    Array.from(staged.files).forEach(f => fd.append('images', f));
    submitBtn.disabled = true;
    loadingOverlay.classList.add('show');
    try {
      const r = await fetch('/setup', { method: 'POST', body: fd });
      const data = await r.json();
      if (!r.ok || !data.ok) {
        alert('Setup failed: ' + (data.error || r.status));
        submitBtn.disabled = false;
        return;
      }
      if (data.failed && data.failed.length) {
        alert(`${data.failed.length} file(s) skipped: ${data.failed.map(f => f.name).join(', ')}`);
      }
      window.location.href = '/';
    } catch (err) {
      alert('Setup failed: ' + err);
      submitBtn.disabled = false;
    } finally {
      loadingOverlay.classList.remove('show');
    }
  });

  refreshList();

  // ── Tab switching ─────────────────────────────────────────────────────────
  const tabs       = document.querySelectorAll('.src-tab');
  const panelLocal  = document.getElementById('panelLocal');
  const panelHestia = document.getElementById('panelHestia');

  let hestiaLoaded = false;

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const which = tab.dataset.tab;
      panelLocal.hidden  = which !== 'local';
      panelHestia.hidden = which !== 'hestia';
      if (which === 'hestia' && !hestiaLoaded) {
        hestiaLoaded = true;
        loadHestiaScans();
      }
    });
  });

  // ── HESTIA inline logic ───────────────────────────────────────────────────
  const hestiaLoading      = document.getElementById('hestiaLoading');
  const hestiaError        = document.getElementById('hestiaError');
  const hestiaList         = document.getElementById('hestiaList');
  const hestiaNamingCard   = document.getElementById('hestiaNamingCard');
  const hestiaDsName       = document.getElementById('hestiaDsName');
  const hestiaNamingCancel = document.getElementById('hestiaNamingCancel');
  const hestiaNamingConfirm= document.getElementById('hestiaNamingConfirm');
  const hestiaDlCard       = document.getElementById('hestiaDlCard');
  const hestiaDlSub        = document.getElementById('hestiaDlSub');
  const hestiaDlFill       = document.getElementById('hestiaDlFill');

  function fmtDate(iso) {
    try { return new Date(iso).toLocaleString(); } catch { return iso; }
  }

  async function loadHestiaScans() {
    hestiaLoading.style.display = 'flex';
    hestiaError.style.display   = 'none';
    hestiaList.style.display    = 'none';
    try {
      const r = await fetch('/hestia/scans', { cache: 'no-store' });
      const d = await r.json();
      if (!d.ok) throw new Error(d.error || 'Failed to load scans');
      renderHestiaScans(d.scans || []);
    } catch (e) {
      hestiaLoading.style.display = 'none';
      hestiaError.style.display   = 'block';
      hestiaError.textContent = `Could not reach HESTIA: ${e.message}`;
    }
  }

  function renderHestiaScans(scans) {
    hestiaLoading.style.display = 'none';
    hestiaList.innerHTML = '';
    if (!scans.length) {
      hestiaError.style.display = 'block';
      hestiaError.textContent   = 'No scans found in HESTIA.';
      return;
    }
    for (const s of scans) {
      const li = document.createElement('li');
      li.className = 'hestia-item';
      li.innerHTML = `
        <div class="hestia-icon">📦</div>
        <div class="hestia-meta">
          <div class="hestia-scan-id"></div>
          <div class="hestia-scan-sub"></div>
        </div>
        <button class="hestia-load-btn">Load</button>`;
      li.querySelector('.hestia-scan-id').textContent = s.scan_id;
      li.querySelector('.hestia-scan-sub').textContent =
        `${s.image_count} image${s.image_count === 1 ? '' : 's'} · ${fmtDate(s.timestamp)}`;
      li.querySelector('.hestia-load-btn').addEventListener('click', () => startHestiaLoad(s.scan_id, s.image_count));
      hestiaList.appendChild(li);
    }
    hestiaList.style.display = 'flex';
  }

  // Step 1: show naming card with pre-filled auto name
  function startHestiaLoad(scanId, imageCount) {
    document.querySelectorAll('.hestia-load-btn').forEach(b => b.disabled = true);
    hestiaList.style.display        = 'none';
    hestiaNamingCard.style.display  = 'flex';
    hestiaDsName.value = `scan_${scanId.slice(0, 8)}`;
    hestiaDsName.focus();
    hestiaDsName.select();

    hestiaNamingCancel.onclick = () => {
      hestiaNamingCard.style.display = 'none';
      document.querySelectorAll('.hestia-load-btn').forEach(b => b.disabled = false);
      hestiaList.style.display = 'flex';
    };

    hestiaNamingConfirm.onclick = () => {
      const name = hestiaDsName.value.trim().replace(/[^A-Za-z0-9._\-]/g, '_');
      if (!name) { hestiaDsName.focus(); return; }
      hestiaNamingCard.style.display = 'none';
      _doHestiaDownload(scanId, imageCount, name);
    };

    hestiaDsName.onkeydown = (e) => {
      if (e.key === 'Enter') hestiaNamingConfirm.click();
      if (e.key === 'Escape') hestiaNamingCancel.click();
    };
  }

  // Step 2: actually start the download with the chosen name
  function _doHestiaDownload(scanId, imageCount, datasetName) {
    hestiaDlCard.style.display = 'flex';
    hestiaDlSub.textContent    = 'Starting download…';
    hestiaDlFill.style.width   = '5%';

    fetch('/hestia/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scan_id: scanId, dataset_name: datasetName, model: selectedModel() }),
    })
    .then(r => r.json())
    .then(d => {
      if (!d.ok) { showHestiaError(d.error || 'Load failed'); return; }
      pollHestia(scanId, imageCount, d.dataset);
    })
    .catch(e => showHestiaError(e.message));
  }

  function pollHestia(scanId, imageCount, datasetName) {
    let attempts = 0;
    const t = setInterval(async () => {
      attempts++;
      try {
        const r = await fetch(`/hestia/load/status?scan_id=${encodeURIComponent(scanId)}`, { cache: 'no-store' });
        const d = await r.json();
        if (!d.ok) { clearInterval(t); showHestiaError(d.error || 'Status error'); return; }

        const pct = imageCount > 0 ? Math.min(95, Math.round((d.downloaded / imageCount) * 100)) : 50;
        hestiaDlFill.style.width = pct + '%';
        hestiaDlSub.textContent  = `Downloaded ${d.downloaded}${imageCount ? ' / ' + imageCount : ''} images…`;

        if (d.status === 'done') {
          clearInterval(t);
          hestiaDlFill.style.width = '100%';
          hestiaDlSub.textContent  = `Done! Redirecting…`;
          setTimeout(() => { window.location.href = '/pick'; }, 800);
        } else if (d.status === 'error') {
          clearInterval(t);
          showHestiaError(d.error || 'Download error');
        }
      } catch (_) {
        if (attempts > 120) { clearInterval(t); showHestiaError('Polling timeout'); }
      }
    }, 2000);
  }

  function showHestiaError(msg) {
    hestiaDlCard.style.display      = 'none';
    hestiaNamingCard.style.display  = 'none';
    hestiaError.style.display       = 'block';
    hestiaError.textContent         = `Error: ${msg}`;
    document.querySelectorAll('.hestia-load-btn').forEach(b => b.disabled = false);
    hestiaList.style.display        = 'flex';
  }
})();
