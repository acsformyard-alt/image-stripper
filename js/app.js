import { maskUpperRightViaOCR as defaultMaskUpperRightViaOCR } from './lama.js';

// UI glue: image selection pills, spinner, per-image progress, sequential processing
let _statusEl, _inpaintOne;
let _ocrMaskBuilder = defaultMaskUpperRightViaOCR;

let _avgInferMs = 1200; // rolling average for smoother progress
const PROMPT_UPPER_RIGHT = /upper\s*right/i;
const PROMPT_REMOVE_TEXT = /(remove|erase|clean).*(text|number|digit)/i;
const DEFAULT_OCR_ZONE = { x0: 0.6, y0: 0.0, x1: 1.0, y1: 0.4 };

export function uiInit({ inpaintOne, statusEl, ocrMask }) {
  _statusEl = statusEl;
  _inpaintOne = inpaintOne;
  if (typeof ocrMask === 'function') {
    _ocrMaskBuilder = ocrMask;
  } else {
    _ocrMaskBuilder = defaultMaskUpperRightViaOCR;
  }

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

      const promptValue = document.getElementById('prompt')?.value || '';
      const wantsUpperRight = PROMPT_UPPER_RIGHT.test(promptValue);
      PROMPT_UPPER_RIGHT.lastIndex = 0;
      const wantsRemoveText = PROMPT_REMOVE_TEXT.test(promptValue);
      PROMPT_REMOVE_TEXT.lastIndex = 0;
      const shouldUseOCR = wantsUpperRight && wantsRemoveText && typeof _ocrMaskBuilder === 'function';

      let externalMask;
      if (shouldUseOCR) {
        stageEl.textContent = 'detect text';
        setProgress(bar, pctEl, 0.15);
        try {
          const srcCanvas = bitmapToCanvas(bitmaps[i].bmp);
          externalMask = await _ocrMaskBuilder(srcCanvas, { zone: DEFAULT_OCR_ZONE, dilatePx: 10 });
          setProgress(bar, pctEl, 0.25);
        } catch (err) {
          console.error('OCR mask generation failed, falling back to rectangle.', err);
          externalMask = undefined;
          setProgress(bar, pctEl, 0.15);
        }
      } else {
        setProgress(bar, pctEl, 0.10);
      }

      let running = true;
      let rafId;
      const start = performance.now();
      const est   = Math.max(300, _avgInferMs);
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
        ({ canvas: result, timings } = await _inpaintOne(bitmaps[i].bmp, { externalMask }));
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

function bitmapToCanvas(bmp) {
  const c = document.createElement('canvas');
  c.width = bmp.width;
  c.height = bmp.height;
  c.getContext('2d').drawImage(bmp, 0, 0);
  return c;
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
