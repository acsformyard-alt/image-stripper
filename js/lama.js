// LaMa inpainting via onnxruntime-web with robust I/O detection and correct normalization.

let session = null;
let _executionProviders = ['wasm']; // stability default; you can switch to ['webgpu','wasm']
let _logger = (m)=>console.log('[lama]', m);

// Detected / configured characteristics
let _io = { image: 'image', mask: 'mask', output: null };
let _target = 512;        // set from model if static [1,3,H,W]
let _assumeBGR = false;   // set true if your export expects BGR input instead of RGB

export function setExecutionProviders(list) {
  if (Array.isArray(list) && list.length) _executionProviders = list;
}
export function setLogger(fn) { if (typeof fn === 'function') _logger = fn; }
export function setAssumeBGR(v) { _assumeBGR = !!v; }

export async function initLamaFromBuffer(bufferUint8, executionProviders = _executionProviders) {
  const ep = pickEP(executionProviders);
  _logger(`creating session; requested EP=${JSON.stringify(ep)}`);
  session = await ort.InferenceSession.create(bufferUint8, {
    executionProviders: ep,
    graphOptimizationLevel: 'all',
  });
  session._modelBytes = bufferUint8; // keep for potential rebuild
  inspectModel();
}

function pickEP(req) {
  // We can't rely on ort.getAvailableExecutionProvider(). Instead:
  // - 'webgpu' is usable only if (a) browser exposes navigator.gpu AND
  //   (b) you loaded the WebGPU entry (ort.webgpu.min.js exposes ort.webgpu).
  // - 'wasm' is always valid if you loaded any ort.*.js and have wasm binaries.
  const out = [];
  for (const p of req) {
    if (p === 'webgpu') {
      if (typeof navigator !== 'undefined' && 'gpu' in navigator && (ort && ort.webgpu)) {
        out.push('webgpu');
      }
      continue;
    }
    if (p === 'wasm') {
      out.push('wasm');
      continue;
    }
  }
  return out.length ? out : ['wasm'];
}


function inspectModel(){
  if(!session) return;
  const ins = session.inputNames || [];
  const outs = session.outputNames || [];
  const md = session.inputMetadata || {};
  _logger(`inputs: ${JSON.stringify(ins)} outputs: ${JSON.stringify(outs)}`);

  // choose inputs by channels (NCHW): 3ch→image, 1ch→mask
  let img = ins.find(n => md[n]?.dimensions?.[1] === 3) || ins[0];
  let msk = ins.find(n => md[n]?.dimensions?.[1] === 1) || ins[1] || ins[0];

  _io.image = img || 'image';
  _io.mask  = msk || 'mask';
  _io.output = outs[0] || null;

  // pick target size if static [N,C,H,W]
  const dims = md[_io.image]?.dimensions || [];
  if (Number.isFinite(dims[2]) && dims[2] === dims[3]) _target = dims[2];

  _logger(`I/O chosen -> image="${_io.image}" mask="${_io.mask}" out="${_io.output}" target=${_target}`);
}

// Tune if your corner text area differs
const UPPER_RIGHT_FRACTION = { w: 0.28, h: 0.24 };

/** Process a single bitmap and return { canvas, timings: {pre, infer, post} } */
export async function inpaintUpperRightOne(bmp) {
  if (!session) throw new Error('Model not initialized. Pick the .onnx first.');
  const t0 = performance.now();

  const srcCanvas  = drawBitmapToCanvas(bmp);
  const maskCanvas = buildUpperRightMask(srcCanvas, UPPER_RIGHT_FRACTION);

  // Preprocess to _target×_target
  const target = _target;
  const { prep, invMap }   = letterbox(srcCanvas, target, target);
  const { prep: prepMask } = letterbox(maskCanvas, target, target);

  const imgData = prep.getContext('2d').getImageData(0, 0, target, target).data;
  const mData   = prepMask.getContext('2d').getImageData(0, 0, target, target).data;

  const imgTensor  = new ort.Tensor('float32', new Float32Array(1*3*target*target), [1,3,target,target]);
  const maskTensor = new ort.Tensor('float32', new Float32Array(1*1*target*target), [1,1,target,target]);

  // NHWC -> NCHW, normalize to [-1,1]; optional BGR swap
  for (let i=0, px=0; i<imgData.length; i+=4, px++){
    let r = imgData[i] / 255, g = imgData[i+1] / 255, b = imgData[i+2] / 255;
    if (_assumeBGR) { const t = r; r = b; b = t; }  // swap to BGR if needed
    r = r*2 - 1; g = g*2 - 1; b = b*2 - 1;          // [-1,1]
    const y = Math.floor(px/target), x = px % target, o = y*target + x;
    imgTensor.data[0*target*target + o] = r;
    imgTensor.data[1*target*target + o] = g;
    imgTensor.data[2*target*target + o] = b;
  }
  // Mask: 1 = hole (fill), 0 = keep
  for (let i=0, px=0; i<mData.length; i+=4, px++){
    const v = (mData[i] | mData[i+1] | mData[i+2] | mData[i+3]) ? 1.0 : 0.0;
    const y = Math.floor(px/target), x = px % target, o = y*target + x;
    maskTensor.data[o] = v;
  }

  // Debug minima/maxima
  // _logger(`img min/max: ${minmax(imgTensor.data)} mask sum: ${sum(maskTensor.data)}`);

  const tPre = performance.now();

  // Run with detected names
  const feeds = {};
  feeds[_io.image] = imgTensor;
  feeds[_io.mask]  = maskTensor;

  const results = await session.run(feeds);
  const tInfer = performance.now();

  const outName = _io.output || (session.outputNames ? session.outputNames[0] : Object.keys(results)[0]);
  const out = results[outName]; // [1,3,H,W] often in [-1,1]

  // Postprocess ([-1,1] -> [0,1]) and map back to original aspect
  const outSquare = nchwToCanvas(out.data, target, target);
  const outCanvas = invMap(outSquare);
  const tPost = performance.now();

  return {
    canvas: outCanvas,
    timings: {
      pre:   tPre  - t0,
      infer: tInfer - tPre,
      post:  tPost - tInfer,
    }
  };
}

/* ---------- helpers ---------- */
function drawBitmapToCanvas(bmp){ const c=document.createElement('canvas'); c.width=bmp.width; c.height=bmp.height; c.getContext('2d').drawImage(bmp,0,0); return c; }
function buildUpperRightMask(canvas, frac){ const c=document.createElement('canvas'); c.width=canvas.width; c.height=canvas.height; const g=c.getContext('2d'); const rw=Math.round(canvas.width*frac.w); const rh=Math.round(canvas.height*frac.h); const rx=canvas.width-rw; g.clearRect(0,0,c.width,c.height); g.fillStyle='#fff'; g.fillRect(rx,0,rw,rh); return c; }
function letterbox(srcCanvas,W,H){ const sw=srcCanvas.width, sh=srcCanvas.height; const scale=Math.min(W/sw,H/sh); const nw=Math.round(sw*scale), nh=Math.round(sh*scale); const dx=Math.floor((W-nw)/2), dy=Math.floor((H-nh)/2); const c=document.createElement('canvas'); c.width=W; c.height=H; const g=c.getContext('2d'); g.fillStyle='#000'; g.fillRect(0,0,W,H); g.drawImage(srcCanvas,0,0,sw,sh,dx,dy,nw,nh); const inv=(square)=>{ const tmp=document.createElement('canvas'); tmp.width=sw; tmp.height=sh; const tg=tmp.getContext('2d'); const crop=document.createElement('canvas'); crop.width=nw; crop.height=nh; crop.getContext('2d').drawImage(square,dx,dy,nw,nh,0,0,nw,nh); tg.drawImage(crop,0,0,nw,nh,0,0,sw,sh); return tmp; }; return { prep:c, invMap:inv }; }
function nchwToCanvas(data,H,W){ const c=document.createElement('canvas'); c.width=W; c.height=H; const g=c.getContext('2d'); const img=g.createImageData(W,H); const plane=H*W; for(let y=0;y<H;y++){ for(let x=0;x<W;x++){ const o=y*W+x; // [-1,1] -> [0,1]
      const r=clamp01((data[0*plane+o]+1)*0.5); const gg=clamp01((data[1*plane+o]+1)*0.5); const b=clamp01((data[2*plane+o]+1)*0.5);
      const i=o*4; img.data[i]=r*255|0; img.data[i+1]=gg*255|0; img.data[i+2]=b*255|0; img.data[i+3]=255; } } g.putImageData(img,0,0); return c; }
const clamp01 = v => v<0?0:v>1?1:v;
// const minmax = (a)=>{ let mi=Infinity, ma=-Infinity; for(const v of a){ if(v<mi)mi=v; if(v>ma)ma=v; } return [mi,ma]; }
// const sum = (a)=>{ let s=0; for(const v of a) s+=v; return s; }

