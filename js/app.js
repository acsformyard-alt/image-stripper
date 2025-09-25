// UI glue: image selection pills, spinner, progress, batch inpaint
let _statusEl, _inpaintBatch;

export function uiInit({ inpaintBatch, statusEl }) {
  _statusEl = statusEl;
  _inpaintBatch = inpaintBatch;

  const $files   = document.getElementById('fileInput');
  const $process = document.getElementById('processBtn');
  const $gallery = document.getElementById('gallery');

  // Update images pill on selection
  $files.addEventListener('change', () => updateImagesInfo($files.files));

  async function handleProcess() {
    const modelSelected = document.getElementById('modelName').textContent !== 'No model selected';
    if (!modelSelected) { _statusEl.textContent = 'Load the model first.'; return; }
    const files = $files.files;
    if (!files || files.length === 0) { _statusEl.textContent = 'Choose images first.'; return; }

    setBusy(true, `Processing 0/${files.length}…`);

    // Read images -> bitmaps
    const bitmaps = [];
    for (const f of files) {
      const url = URL.createObjectURL(f);
      const img = new Image(); img.src = url; await img.decode();
      const bmp = await createImageBitmap(img);
      URL.revokeObjectURL(url);
      bitmaps.push({ name: f.name, bmp });
    }

    // Run sequentially so we can show per-item progress
    $gallery.innerHTML = '';
    const outs = [];
    for (let i = 0; i < bitmaps.length; i++) {
      _statusEl.textContent = `Processing ${i+1}/${bitmaps.length}…`;
      const res = await _inpaintBatch([bitmaps[i].bmp]); // one at a time
      outs.push(res[0]);
    }

    // Render
    outs.forEach((outCanvas, i) => {
      const { canv, btn } = addTile($files.files[i].name, $gallery);
      canv.width = outCanvas.width; canv.height = outCanvas.height;
      canv.getContext('2d').drawImage(outCanvas, 0, 0);
      btn.onclick = () => downloadCanvas(canv, $files.files[i].name);
    });

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
  wrap.appendChild(head); wrap.appendChild(canv); container.appendChild(wrap);
  return { canv, btn, wrap };
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
