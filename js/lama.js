// LaMa inpainting via onnxruntime-web
// Fixed prompt: mask upper-right rectangle and inpaint.

let session = null;
let _executionProviders = ['webgpu', 'wasm'];

export function setExecutionProviders(list) {
  if (Array.isArray(list) && list.length) _executionProviders = list;
}

/** Initialize from bytes (works with local file input; no 25MB page limit). */
export async function initLamaFromBuffer(bufferUint8, executionProviders = _executionProviders) {
  session = await ort.InferenceSession.create(bufferUint8, {
    executionProviders,
    graphOptimizationLevel: 'all',
  });
}

const UPPER_RIGHT_FRACTION = { w: 0.28, h: 0.24 }; // tweak if your stamp area differs

export async function inpaintUpperRightBatch(bitmaps) {
  if (!session) throw new Error('Model not initialized. Pick the .onnx first.');
  const outs = [];
  for (const bmp of bitmaps) {
    const srcCanvas  = drawBitmapToCanvas(bmp);
    const maskCanvas = buildUpperRightMask(srcCanvas, UPPER_RIGHT_FRACTION);
    const out = await runLamaInpaint(srcCanvas, maskCanvas);
    outs.push(out);
  }
  return outs;
}

async function runLamaInpaint(srcCanvas, maskCanvas) {
  const target = 512; // change if your ONNX expects another size
  const { prep, invMap }     = letterbox(srcCanvas, target, target);
  const { prep: prepMask }   = letterbox(maskCanvas, target, target);

  const imgData = prep.getContext('2d').getImageData(0, 0, target, target).data;
  const mData   = prepMask.getContext('2d').getImageData(0, 0, target, target).data;

  const imgTensor  = new ort.Tensor('float32', new Float32Array(1*3*target*target), [1,3,target,target]);
  const maskTensor = new ort.Tensor('float32', new Float32Array(1*1*target*target), [1,1,target,target]);

  // NHWC -> NCHW, normalize 0..1
  for (let i = 0, px = 0; i < imgData.length; i += 4, px++) {
    const r = imgData[i]/255, g = imgData[i+1]/255, b = imgData[i+2]/255;
    const y = Math.floor(px/target), x = px % target, o = y*target + x;
    imgTensor.data[0*target*target + o] = r;
    imgTensor.data[1*target*target + o] = g;
    imgTensor.data[2*target*target + o] = b;
  }
  // Mask: any nonzero -> 1
  for (let i = 0, px = 0; i < mData.length; i += 4, px++) {
    const v = (mData[i] | mData[i+1] | mData[i+2] | mData[i+3]) ? 1.0 : 0.0;
    const y = Math.floor(px/target), x = px % target, o = y*target + x;
    maskTensor.data[o] = v;
  }

  // Adjust names if your export differs
  const feeds = { image: imgTensor, mask: maskTensor };
  const results = await session.run(feeds);
  const outName = session.outputNames ? session.outputNames[0] : Object.keys(results)[0];
  const out = results[outName]; // [1,3,H,W], float32 [0,1]

  const outSquare = nchwToCanvas(out.data, target, target);
  return invMap(outSquare);
}

function drawBitmapToCanvas(bmp) {
  const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
  c.getContext('2d').drawImage(bmp, 0, 0);
  return c;
}
function buildUpperRightMask(canvas, frac) {
  const c = document.createElement('canvas'); c.width = canvas.width; c.height = canvas.height;
  const g = c.getContext('2d');
  const rw = Math.round(canvas.width * frac.w);
  const rh = Math.round(canvas.height * frac.h);
  const rx = canvas.width - rw;
  g.clearRect(0,0,c.width,c.height);
  g.fillStyle = '#fff'; g.fillRect(rx, 0, rw, rh);
  return c;
}
function letterbox(srcCanvas, W, H) {
  const srcW = srcCanvas.width, srcH = srcCanvas.height;
  const scale = Math.min(W/srcW, H/srcH);
  const nw = Math.round(srcW*scale), nh = Math.round(srcH*scale);
  const dx = Math.floor((W-nw)/2), dy = Math.floor((H-nh)/2);
  const c = document.createElement('canvas'); c.width=W; c.height=H;
  const g = c.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0,0,W,H);
  g.drawImage(srcCanvas, 0,0, srcW,srcH, dx,dy, nw,nh);
  const invMap = (squareCanvas) => {
    const tmp = document.createElement('canvas'); tmp.width = srcW; tmp.height = srcH;
    const tg = tmp.getContext('2d');
    const crop = document.createElement('canvas'); crop.width = nw; crop.height = nh;
    crop.getContext('2d').drawImage(squareCanvas, dx, dy, nw, nh, 0, 0, nw, nh);
    tg.drawImage(crop, 0, 0, nw, nh, 0, 0, srcW, srcH);
    return tmp;
  };
  return { prep: c, invMap };
}
function nchwToCanvas(data, H, W) {
  const c = document.createElement('canvas'); c.width=W; c.height=H;
  const g = c.getContext('2d'); const img = g.createImageData(W,H); const plane=H*W;
  for (let y=0; y<H; y++) for (let x=0; x<W; x++) {
    const o=y*W+x; const r=clamp01(data[0*plane+o]); const gg=clamp01(data[1*plane+o]); const b=clamp01(data[2*plane+o]);
    const i=o*4; img.data[i]=r*255|0; img.data[i+1]=gg*255|0; img.data[i+2]=b*255|0; img.data[i+3]=255;
  }
  g.putImageData(img,0,0); return c;
}
const clamp01 = v => v<0?0:v>1?1:v;
