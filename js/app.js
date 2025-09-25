// UI glue: image selection pills, spinner, per-image progress, sequential processing
let _statusEl, _inpaintOne;

let _avgInferMs = 1200; // rolling average for smoother progress

export function uiInit({ inpaintOne, statusEl }) {
  _statusEl = statusEl;
  _inpaintOne = inpaintOne;

  const $files   = document.getElementById('fileInput');
  const $process = document.getElementById('processBtn');
  const $gallery = document.getElementById('gallery');

  $files.addEventListener('change', () => updateImagesInfo($files.files));

  async function handleProcess() {
    const modelSelected = document.getElementById('modelName').textContent !== 'No model selected';
    if (!modelSelected) { _statusEl.textContent = 'Load the model first.'; return; }
    const files = $files.files;
    if (!files || files.length === 0) { _statusEl.textContent = 'Choose images first.'; return; }

    setBusy(true, `Processing 0/${files.length}…`);

    // Decode upfront (fast), then process sequentially
    const bitmaps = [];
    for (const f of files) {
      const url = URL.createObjectURL(f);
      const img = new Image(); img.src = url; await img.decode();
      const bmp = await createImageBitmap(img);
      URL.revokeObjectURL(url);
      bitmaps.push({ name: f.name, bmp });
    }

    $gallery.innerHTML = '';
    const tiles = bitmaps.map(b => addTile(b.name, $gallery));

    for (let i = 0; i < bitmaps.length; i++) {
      const tile = tiles[i];
      const { canv, btn, bar, stageEl, pctEl } = tile;

      stageEl.textContent = 'preprocess';
      setProgress(bar, pctEl, 0.05);
      _statusEl.textContent = `Processing ${i+1}/${bitmaps.length}…`;

      let running = true;
      const start = performance.now();
      const est   = Math.max(300, _avgInferMs);
      let rafId;
      const tick = () => {
        if (!running) return;
        const elapsed = performance.now() - start;
        const frac = Math.min(0.95, (elapsed / est) * 0.85 + 0.10);
        stageEl.textContent = 'inference';
        setProgress(bar, pctEl, frac);
        rafId = requestAnimationFrame(tick);
      };
      rafId = requestAnimationFrame(tick);

      let result, timings;
      try {
        ({ canvas: result, timings } = await _inpaintOne(bitmaps[i].bmp));
      } catch (e) {
        running = false; if (rafId) cancelAnimationFrame(rafId);
        stageEl.textContent = 'error';
        setProgress(bar, pctEl, 1);
        canv.width = 600; canv.height = 50;
        const g = canv.getContext('2d'); g.fillStyle = '#fff'; g.fillText('Error: ' + (e.message || e), 10, 28);
        continue;
      }
      running = false; if (rafId) cancelAnimationFrame(rafId);

      if (timings?.infer) _avgInferMs = 0.7 * _avgInferMs + 0.3 * timings.infer;

      stageEl.textContent = 'postprocess';
      setProgress(bar, pctEl, 0.98);

      canv.width = result.width; canv.height = result.height;
      canv.getContext('2d').drawImage(result, 0, 0);

      setProgress(bar, pctEl, 1.0);
      stageEl.textContent = 'done';
      btn.onclick = () => downloadCanvas(canv, files[i].name);
    }

    _statusEl.textContent = 'Done.';
    setBusy(false);
  }

  $process.addEventListener('click', handleProcess);
}

export function wireImagePicker() {
  const $files = document.getElementById('fileInput');
  updateImagesInfo($files.files);
}

export function setModelLabel(text) {
  const $modelName = document.getElementById('modelName');
  $modelName.textContent = text || 'No model selected';
}

export function setBusy(isBusy, statusText) {
  document.body.classList.toggle('busy', !!isBusy);
  const spin = document.getElementById('globalSpinner');
  if (spin) spin.setAttribute('aria-hidden', isBusy ? 'false' : 'true');
  if (statusText && _statusEl) _statusEl.textContent = statusText;
}

function updateImagesInfo(fileList) {
  const $info = document.getElementById('imagesInfo');
  if (!fileList || fileList.length === 0) { $info.textContent = 'No images'; return; }
  const names = Array.from(fileList).map(f => f.name);
  const preview = names.length > 3 ? names.slice(0,3).join(', ') + ` … (+${names.length-3})` : names.join(', ');
  $info.textContent = `${fileList.length} file(s): ${preview}`;
}

function addTile(name, container) {
  const wrap = document.createElement('div'); wrap.className = 'tile';
  const head = document.createElement('header'); head.innerHTML = `<span class="meta">${name}</span>`;
  const btn = document.createElement('button'); btn.textContent = 'Download'; head.appendChild(btn);
  const canv = document.createElement('canvas');
  const progWrap = document.createElement('div'); progWrap.className = 'progress-wrap';
  const row = document.createElement('div'); row.className = 'progress-row';
  const stage = document.createElement('span'); stage.className = 'stage'; stage.textContent = 'ready';
  const pct = document.createElement('span'); pct.className = 'pct'; pct.textContent = '0%';
  const bar = document.createElement('progress'); bar.max = 1; bar.value = 0;
  row.appendChild(stage); row.appendChild(bar); row.appendChild(pct);
  progWrap.appendChild(row);

  wrap.appendChild(head); wrap.appendChild(canv); wrap.appendChild(progWrap); container.appendChild(wrap);
  return { canv, btn, wrap, bar, stageEl: stage, pctEl: pct };
}

function setProgress(bar, pctEl, v) {
  const clamped = Math.max(0, Math.min(1, v));
  bar.value = clamped;
  pctEl.textContent = Math.round(clamped * 100) + '%';
}

function downloadCanvas(canv, originalName) {
  canv.toBlob(blob => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = originalName.replace(/\.[^.]+$/, '') + '.clean.png';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 500);
  }, 'image/png');
}
