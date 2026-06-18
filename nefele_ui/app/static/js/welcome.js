(() => {
  'use strict';
  const root = document.getElementById('welcomeModel');
  if (!root) return;

  const status = document.getElementById('welcomeModelStatus');
  const opts   = root.querySelectorAll('.welcome-model-opt');

  function setActive(model) {
    opts.forEach(b => {
      const on = b.dataset.model === model;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-checked', on ? 'true' : 'false');
    });
    root.dataset.current = model;
  }

  function flashStatus(text, ok = true) {
    if (!status) return;
    status.textContent = text;
    status.style.color = ok ? '' : '#a8261b';
    if (ok) setTimeout(() => { status.textContent = ''; }, 1500);
  }

  opts.forEach(btn => {
    btn.addEventListener('click', async () => {
      const next = btn.dataset.model;
      if (next === root.dataset.current) return;
      const prev = root.dataset.current;
      setActive(next);
      try {
        const r = await fetch('/welcome/model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: next }),
        });
        const d = await r.json();
        if (!r.ok || !d.ok) {
          setActive(prev);
          flashStatus(d.error || 'Update failed', false);
          return;
        }
        flashStatus('Saved.');
      } catch (e) {
        setActive(prev);
        flashStatus('Network error', false);
      }
    });
  });
})();
