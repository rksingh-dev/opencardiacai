"use client";

import React, { useState, useRef, useEffect } from 'react';
import {
  Activity, Heart, Layers, ImageUp, Play, Trash2,
  Cpu, LayoutDashboard, Download, Zap, FileHeart,
  Eye, EyeOff, SlidersHorizontal, TrendingUp
} from 'lucide-react';
import * as ort from 'onnxruntime-web';
import ThreeDViewer from '../components/ThreeDViewer';

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// ── Types ──────────────────────────────────────────────────
type Status = "idle" | "processing" | "success" | "computing_ef";
type FileMode = 'image' | 'dicom' | '3d_volume' | '4d_cine';

interface Diagnostics {
  rvArea: number;  myoArea: number;  lvArea: number;
  diagnosis: string;  myoLvRatio: number;
  ejectionFraction?: number;
  lvVolMl?: number; myoVolMl?: number; rvVolMl?: number;
}
interface SliceMeta { cols: number; rows: number; sliceCount: number; }

// ── Feature 3: Windowing ───────────────────────────────────
// Applies WW/WL windowing to raw single-channel uint8 b64 bytes
function applyWindowingToB64(b64: string, cols: number, rows: number, wl: number, ww: number): string {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const canvas = document.createElement('canvas');
  canvas.width = cols; canvas.height = rows;
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.createImageData(cols, rows);
  const low = wl - ww / 2;
  for (let i = 0; i < bytes.length; i++) {
    let v = bytes[i] <= low ? 0 : bytes[i] >= low + ww ? 255 : ((bytes[i] - low) / ww) * 255;
    const idx = i * 4;
    imgData.data[idx] = v; imgData.data[idx+1] = v; imgData.data[idx+2] = v; imgData.data[idx+3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}

// Converts raw b64 bytes to full-range dataURL (for inference - no windowing)
function rawB64ToDataUrl(b64: string, cols: number, rows: number): string {
  return applyWindowingToB64(b64, cols, rows, 128, 255);
}

// ── Feature 1: Confidence color map ───────────────────────
function confidenceToRGB(conf: number): [number, number, number] {
  // Blue (low confidence) → Green → Red (high confidence)
  if (conf < 0.5) {
    const t = conf * 2;
    return [0, Math.round(t * 200), Math.round(255 * (1 - t))];
  }
  const t = (conf - 0.5) * 2;
  return [Math.round(t * 255), Math.round(200 * (1 - t)), 0];
}

// ── Feature 6: Wall thickness by 12-sector ray casting ────
function computeWallThickness(imgData: ImageData, size: number): number[] {
  const d = imgData.data;
  let cx = 0, cy = 0, lvPx = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      if (d[i] > 180 && d[i+1] < 100 && d[i+2] < 100 && d[i+3] > 50) {
        cx += x; cy += y; lvPx++;
      }
    }
  }
  if (lvPx < 10) return Array(12).fill(0);
  cx /= lvPx; cy /= lvPx;

  return Array.from({ length: 12 }, (_, s) => {
    const angle = (s / 12) * 2 * Math.PI;
    const cos = Math.cos(angle), sin = Math.sin(angle);
    let start = -1, end = -1;
    for (let r = 3; r < size / 2; r++) {
      const px = Math.round(cx + r * cos), py = Math.round(cy + r * sin);
      if (px < 0 || px >= size || py < 0 || py >= size) break;
      const idx = (py * size + px) * 4;
      const isMyo = d[idx] < 50 && d[idx+1] > 150 && d[idx+2] > 90 && d[idx+3] > 50;
      if (isMyo && start === -1) start = r;
      if (!isMyo && start !== -1) { end = r; break; }
    }
    if (start !== -1 && end === -1) end = start + 1;
    return start !== -1 ? Math.max(0, end - start) : 0;
  });
}

// ── Feature 6: Wall Thickness Polar SVG Chart ─────────────
function WallThicknessChart({ thicknesses, pixdim }: { thicknesses: number[], pixdim: number[] }) {
  const S = 130, c = S / 2, maxR = S / 2 - 16;
  const max = Math.max(...thicknesses, 1);
  const vox = (pixdim[0] + pixdim[1]) / 2 || 1;
  return (
    <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
      <svg width={S} height={S} viewBox={`0 0 ${S} ${S}`}>
        <circle cx={c} cy={c} r={maxR * 0.5} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth={0.5} />
        <circle cx={c} cy={c} r={maxR}       fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth={0.5} />
        {thicknesses.map((t, s) => {
          const a1 = (s / 12) * 2 * Math.PI - Math.PI / 2;
          const a2 = ((s + 1) / 12) * 2 * Math.PI - Math.PI / 2;
          const r = (t / max) * maxR;
          const pts = [
            [c + Math.cos(a1) * 4,  c + Math.sin(a1) * 4],
            [c + Math.cos(a1) * r,  c + Math.sin(a1) * r],
            [c + Math.cos(a2) * r,  c + Math.sin(a2) * r],
            [c + Math.cos(a2) * 4,  c + Math.sin(a2) * 4],
          ].map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
          const hue = Math.round((1 - t / max) * 200);
          return <polygon key={s} points={pts} fill={`hsl(${hue},80%,50%)`} opacity={0.85} stroke="rgba(0,0,0,0.4)" strokeWidth={0.5} />;
        })}
        <circle cx={c} cy={c} r={4} fill="#333" />
      </svg>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.2rem 1rem' }}>
        {thicknesses.map((t, i) => (
          <div key={i} style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
            <div style={{ width: 8, height: 8, background: `hsl(${Math.round((1 - t/max)*200)},80%,50%)`, flexShrink: 0 }} />
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              Sec {i+1}: <span style={{ color: 'var(--text-secondary)' }}>{(t * vox).toFixed(1)}mm</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Feature 5: Cardiac Phase SVG Line Chart ────────────────
function PhaseChart({ lvAreas, dIdx, sIdx }: { lvAreas: number[], dIdx: number, sIdx: number }) {
  if (lvAreas.length < 2) return null;
  const W = 260, H = 80, pad = 8;
  const maxA = Math.max(...lvAreas), minA = Math.min(...lvAreas), range = maxA - minA || 1;
  const pts = lvAreas.map((a, i) => {
    const x = pad + (i / (lvAreas.length - 1)) * (W - pad * 2);
    const y = pad + (1 - (a - minA) / range) * (H - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const cx = (dIdx / (lvAreas.length - 1)) * (W - pad * 2) + pad;
  const cy = pad + (1 - (lvAreas[dIdx] - minA) / range) * (H - pad * 2);
  const sx = (sIdx / (lvAreas.length - 1)) * (W - pad * 2) + pad;
  const sy = pad + (1 - (lvAreas[sIdx] - minA) / range) * (H - pad * 2);
  return (
    <div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
        <polyline points={pts} fill="none" stroke="#3b82f6" strokeWidth={1.5} strokeLinejoin="round" />
        <line x1={cx} y1={0} x2={cx} y2={H} stroke="#10b981" strokeWidth={0.5} strokeDasharray="2,2" />
        <line x1={sx} y1={0} x2={sx} y2={H} stroke="#ef4444" strokeWidth={0.5} strokeDasharray="2,2" />
        <circle cx={cx} cy={cy} r={4} fill="#10b981" />
        <circle cx={sx} cy={sy} r={4} fill="#ef4444" />
        <text x={cx+4} y={cy-4} fontSize={8} fill="#10b981">ED</text>
        <text x={sx+4} y={sy-4} fontSize={8} fill="#ef4444">ES</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
        <span><span style={{ color: '#10b981' }}>●</span> End-Diastole (frame {dIdx+1})</span>
        <span><span style={{ color: '#ef4444' }}>●</span> End-Systole (frame {sIdx+1})</span>
        <span><span style={{ color: '#3b82f6' }}>—</span> LV Area Curve</span>
      </div>
    </div>
  );
}

// ── Core Segmentation Engine ───────────────────────────────
// Features 1 (confidence) + 2 (multi-slice avg) + 6 (wall thickness)
// Accepts 1-3 dataUrls; averages output tensors before argmax
async function segmentDataUrls(
  dataUrls: string[],
  session: ort.InferenceSession
): Promise<{
  rvCount: number; myoCount: number; lvCount: number;
  maskDataUrl: string; confidenceMapUrl: string; wallThicknesses: number[];
}> {
  const T = 224, TT = T * T;
  const allOutputs: Float32Array[] = [];
  let isNHWC = false, numClasses = 4;

  for (const url of dataUrls) {
    const img = new Image();
    await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = rej; img.src = url; });
    const hc = document.createElement('canvas'); hc.width = T; hc.height = T;
    const hctx = hc.getContext('2d', { willReadFrequently: true })!;
    hctx.drawImage(img, 0, 0, T, T);
    const raw = hctx.getImageData(0, 0, T, T).data;
    const float = new Float32Array(TT);
    let sum = 0;
    for (let i = 0; i < TT; i++) {
      const g = (raw[i*4]*0.299 + raw[i*4+1]*0.587 + raw[i*4+2]*0.114) / 255;
      float[i] = g; sum += g;
    }
    const mean = sum / TT;
    let v = 0; float.forEach(x => v += (x - mean) ** 2);
    const std = Math.sqrt(v / TT) || 1;
    for (let i = 0; i < TT; i++) float[i] = (float[i] - mean) / std;

    const tensor = new ort.Tensor('float32', float, [1, 1, T, T]);
    const out = (await session.run({ [session.inputNames[0]]: tensor }))[session.outputNames[0]];
    if (allOutputs.length === 0) {
      if (out.dims.length === 4) {
        if (out.dims[3] <= 10) { isNHWC = true; numClasses = out.dims[3]; }
        else numClasses = out.dims[1];
      }
    }
    allOutputs.push(out.data as Float32Array);
  }

  // Feature 2: Average logits across slices
  const avg = new Float32Array(allOutputs[0].length);
  for (let i = 0; i < avg.length; i++) {
    let s = 0; for (const o of allOutputs) s += o[i]; avg[i] = s / allOutputs.length;
  }

  // Render mask + Feature 1: confidence heatmap
  const mc = document.createElement('canvas'); mc.width = T; mc.height = T;
  const mctx = mc.getContext('2d')!; const mData = mctx.createImageData(T, T);
  const cc = document.createElement('canvas'); cc.width = T; cc.height = T;
  const cctx = cc.getContext('2d')!; const cData = cctx.createImageData(T, T);
  const COLORS = [[0,0,0,0],[59,130,246,160],[16,185,129,160],[239,68,68,160]];
  let rv = 0, myo = 0, lv = 0;

  for (let y = 0; y < T; y++) {
    for (let x = 0; x < T; x++) {
      const px = y * T + x;
      let cls = 0, mx = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        const val = isNHWC ? avg[px*numClasses+c] : avg[c*TT+px];
        if (val > mx) { mx = val; cls = c; }
      }
      // Numerically stable softmax confidence
      let se = 0;
      for (let c = 0; c < numClasses; c++) {
        se += Math.exp((isNHWC ? avg[px*numClasses+c] : avg[c*TT+px]) - mx);
      }
      const conf = 1 / se;

      if (cls === 1) rv++; if (cls === 2) myo++; if (cls === 3) lv++;
      const [r, g, b, a] = COLORS[cls] ?? [0,0,0,0];
      const i4 = px * 4;
      mData.data[i4]=r; mData.data[i4+1]=g; mData.data[i4+2]=b; mData.data[i4+3]=a;
      const [cr, cg, cb] = confidenceToRGB(conf);
      cData.data[i4]=cr; cData.data[i4+1]=cg; cData.data[i4+2]=cb; cData.data[i4+3]=200;
    }
  }
  mctx.putImageData(mData, 0, 0);
  cctx.putImageData(cData, 0, 0);

  // Feature 6: wall thickness from mask
  const fullData = mctx.getImageData(0, 0, T, T);
  const wallThicknesses = computeWallThickness(fullData, T);

  return { rvCount: rv, myoCount: myo, lvCount: lv, maskDataUrl: mc.toDataURL('image/png'), confidenceMapUrl: cc.toDataURL('image/png'), wallThicknesses };
}

// Lightweight LV-only counter (for EF per-frame loop)
async function countLV(dataUrl: string, session: ort.InferenceSession): Promise<number> {
  const T = 224, TT = T * T;
  const img = new Image();
  await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = rej; img.src = dataUrl; });
  const hc = document.createElement('canvas'); hc.width = T; hc.height = T;
  const hctx = hc.getContext('2d', { willReadFrequently: true })!;
  hctx.drawImage(img, 0, 0, T, T);
  const raw = hctx.getImageData(0, 0, T, T).data;
  const float = new Float32Array(TT);
  let sum = 0;
  for (let i = 0; i < TT; i++) {
    const g = (raw[i*4]*0.299 + raw[i*4+1]*0.587 + raw[i*4+2]*0.114) / 255;
    float[i] = g; sum += g;
  }
  const mean = sum / TT; let vv = 0; float.forEach(x => vv += (x - mean) ** 2);
  const std = Math.sqrt(vv / TT) || 1;
  for (let i = 0; i < TT; i++) float[i] = (float[i] - mean) / std;
  const tensor = new ort.Tensor('float32', float, [1, 1, T, T]);
  const out = (await session.run({ [session.inputNames[0]]: tensor }))[session.outputNames[0]];
  const outData = out.data as Float32Array;
  let isNHWC = false, nc = 4;
  if (out.dims.length === 4) { if (out.dims[3] <= 10) { isNHWC = true; nc = out.dims[3]; } else nc = out.dims[1]; }
  let lv = 0;
  for (let px = 0; px < TT; px++) {
    let cls = 0, mx = -Infinity;
    for (let c = 0; c < nc; c++) {
      const v = isNHWC ? outData[px*nc+c] : outData[c*TT+px];
      if (v > mx) { mx = v; cls = c; }
    }
    if (cls === 3) lv++;
  }
  return lv;
}

function getDiagnosis(myo: number, lv: number, rv: number, ef?: number) {
  if (lv < 50 || myo < 50) return { diagnosis: "Inconclusive (Incomplete View)", ratio: 0 };
  const ratio = myo / lv;
  let diagnosis = "Normal Cardiac Structure";
  if      (ef !== undefined && ef < 35) diagnosis = "Severe Systolic Dysfunction";
  else if (ef !== undefined && ef < 50) diagnosis = "Possible Dilated Cardiomyopathy (DCM)";
  else if (ratio > 1.8)                 diagnosis = "Possible Hypertrophic Cardiomyopathy (HCM)";
  else if (ratio < 0.6)                 diagnosis = "Possible Dilated Cardiomyopathy (DCM)";
  else if (rv > lv * 2.5)              diagnosis = "Abnormal Right Ventricle (ARV)";
  return { diagnosis, ratio };
}

// ══════════════════════════════════════════════════════════
//  MAIN COMPONENT
// ══════════════════════════════════════════════════════════
export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [status, setStatus]               = useState<Status>("idle");
  const [session, setSession]             = useState<ort.InferenceSession | null>(null);
  const [engineReady, setEngineReady]     = useState(false);
  const [metrics, setMetrics]             = useState<Diagnostics | null>(null);

  // Medical volume state
  const [allSlices, setAllSlices]     = useState<string[]>([]);
  const [sliceMeta, setSliceMeta]     = useState<SliceMeta | null>(null);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [fileMode, setFileMode]       = useState<FileMode>('image');
  const [pixdim, setPixdim]           = useState<number[]>([1, 1, 1]);

  // Feature 3: Windowing
  const [windowLevel, setWindowLevel] = useState(128);
  const [windowWidth, setWindowWidth] = useState(255);

  // Feature 1: Confidence toggle
  const [showConfidence, setShowConfidence]     = useState(false);
  const [maskDataUrl, setMaskDataUrl]           = useState<string | null>(null);
  const [confidenceMapUrl, setConfidenceMapUrl] = useState<string | null>(null);

  // Feature 5: EF & phase chart
  const [efProgress, setEfProgress]   = useState(0);
  const [lvAreaCurve, setLvAreaCurve] = useState<number[]>([]);
  const [diastoleIdx, setDiastoleIdx] = useState(-1);
  const [systoleIdx, setSystoleIdx]   = useState(-1);

  // Feature 6: Wall thickness
  const [wallThicknesses, setWallThicknesses] = useState<number[]>([]);

  // 3D rendering state
  const [threeDData, setThreeDData] = useState<{ points: Float32Array, colors: Float32Array } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef    = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    ort.InferenceSession.create('/cine_model.onnx', { executionProviders: ['wasm'] })
      .then(s => { setSession(s); setEngineReady(true); })
      .catch(e => console.error("Model load error:", e));
  }, []);

  // Feature 3: Re-apply windowing when sliders change
  useEffect(() => {
    if (allSlices.length > 0 && sliceMeta) {
      setSelectedImage(applyWindowingToB64(allSlices[currentSlice], sliceMeta.cols, sliceMeta.rows, windowLevel, windowWidth));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [windowLevel, windowWidth]);

  const clearCanvas = () => {
    if (canvasRef.current) {
      canvasRef.current.getContext('2d')!.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const drawOnCanvas = (url: string) => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    canvas.width = 224; canvas.height = 224;
    const img = new Image(); img.onload = () => ctx.drawImage(img, 0, 0); img.src = url;
  };

  const resetState = () => {
    setSelectedImage(null); setMaskDataUrl(null); setConfidenceMapUrl(null);
    setStatus("idle"); setMetrics(null); setAllSlices([]); setSliceMeta(null);
    setCurrentSlice(0); setFileMode('image'); setPixdim([1,1,1]);
    setWindowLevel(128); setWindowWidth(255); setShowConfidence(false);
    setLvAreaCurve([]); setDiastoleIdx(-1); setSystoleIdx(-1);
    setWallThicknesses([]); setEfProgress(0); clearCanvas();
    setThreeDData(null);
  };

  const loadFile = async (file: File) => {
    resetState();
    const fname = file.name.toLowerCase();

    if (fname.endsWith('.dcm') || fname.endsWith('.nii') || fname.endsWith('.nii.gz')) {
      setStatus('processing');
      const form = new FormData(); form.append('file', file);
      try {
        const res  = await fetch('/api/dicom', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        const { cols, rows, sliceCount, mode, slices, pixdim: pd } = data;
        setSliceMeta({ cols, rows, sliceCount });
        setFileMode(mode as FileMode);
        setAllSlices(slices);
        if (pd) setPixdim(pd);
        const mid = Math.floor(sliceCount / 2);
        setCurrentSlice(mid);
        setSelectedImage(applyWindowingToB64(slices[mid], cols, rows, 128, 255));
        setStatus('idle');
      } catch (err) { console.error(err); alert("Failed to parse medical image."); setStatus('idle'); }
      return;
    }

    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = e => { setSelectedImage(e.target?.result as string); setFileMode('image'); };
    reader.readAsDataURL(file);
  };

  const onSliceChange = (idx: number) => {
    if (!sliceMeta || !allSlices.length) return;
    setCurrentSlice(idx);
    setSelectedImage(applyWindowingToB64(allSlices[idx], sliceMeta.cols, sliceMeta.rows, windowLevel, windowWidth));
    setMaskDataUrl(null); setConfidenceMapUrl(null);
    setMetrics(null); setWallThicknesses([]); setShowConfidence(false);
    clearCanvas(); setStatus('idle');
    setThreeDData(null);
  };

  // Features 1+2+6: Segment current slice (with multi-slice averaging)
  const runSegment = async () => {
    if (!selectedImage || !session) return;
    setStatus('processing');
    try {
      let dataUrls: string[];

      if (allSlices.length > 0 && sliceMeta) {
        // Feature 2: collect up to 3 adjacent raw slices → convert to dataUrls (without windowing, for pure inference)
        const prev = Math.max(0, currentSlice - 1);
        const next = Math.min(allSlices.length - 1, currentSlice + 1);
        const b64Set = [...new Set([allSlices[prev], allSlices[currentSlice], allSlices[next]])];
        dataUrls = b64Set.map(b => rawB64ToDataUrl(b, sliceMeta.cols, sliceMeta.rows));
      } else {
        dataUrls = [selectedImage];
      }

      const result = await segmentDataUrls(dataUrls, session);
      setMaskDataUrl(result.maskDataUrl);
      setConfidenceMapUrl(result.confidenceMapUrl);
      setWallThicknesses(result.wallThicknesses);  // Feature 6
      setShowConfidence(false);
      drawOnCanvas(result.maskDataUrl);

      // Feature 4: Volume in ml
      const sf = sliceMeta ? (sliceMeta.cols / 224) * (sliceMeta.rows / 224) : 1;
      const vv = pixdim[0] * pixdim[1] * pixdim[2];
      const toMl = (px: number) => parseFloat(((px * sf * vv) / 1000).toFixed(2));

      const { diagnosis, ratio } = getDiagnosis(result.myoCount, result.lvCount, result.rvCount);
      setMetrics({
        rvArea: result.rvCount, myoArea: result.myoCount, lvArea: result.lvCount,
        diagnosis, myoLvRatio: ratio,
        lvVolMl: toMl(result.lvCount), myoVolMl: toMl(result.myoCount), rvVolMl: toMl(result.rvCount),
      });
      setStatus('success');
    } catch (err) { console.error(err); setStatus('idle'); }
  };

  // Feature 5: EF calculation across all time frames
  const runEF = async () => {
    if (!session || !allSlices.length || !sliceMeta) return;
    setStatus('computing_ef'); setEfProgress(0);
    const areas: number[] = [];
    try {
      for (let i = 0; i < allSlices.length; i++) {
        const url = rawB64ToDataUrl(allSlices[i], sliceMeta.cols, sliceMeta.rows);
        areas.push(await countLV(url, session));
        setEfProgress(Math.round(((i + 1) / allSlices.length) * 100));
      }
      setLvAreaCurve(areas);

      let maxLv = -Infinity, minLv = Infinity, dIdx = 0, sIdx = 0;
      areas.forEach((a, i) => {
        if (a > maxLv) { maxLv = a; dIdx = i; }
        if (a < minLv) { minLv = a; sIdx = i; }
      });
      setDiastoleIdx(dIdx); setSystoleIdx(sIdx);
      const ef = maxLv > 0 ? Math.round(((maxLv - minLv) / maxLv) * 100) : 0;

      // Full segmentation on diastole frame
      const url = rawB64ToDataUrl(allSlices[dIdx], sliceMeta.cols, sliceMeta.rows);
      const result = await segmentDataUrls([url], session);
      setCurrentSlice(dIdx);
      setSelectedImage(applyWindowingToB64(allSlices[dIdx], sliceMeta.cols, sliceMeta.rows, windowLevel, windowWidth));
      setMaskDataUrl(result.maskDataUrl);
      setConfidenceMapUrl(result.confidenceMapUrl);
      setWallThicknesses(result.wallThicknesses);
      setShowConfidence(false);
      drawOnCanvas(result.maskDataUrl);

      const sf = (sliceMeta.cols / 224) * (sliceMeta.rows / 224);
      const vv = pixdim[0] * pixdim[1] * pixdim[2];
      const toMl = (px: number) => parseFloat(((px * sf * vv) / 1000).toFixed(2));

      const { diagnosis, ratio } = getDiagnosis(result.myoCount, result.lvCount, result.rvCount, ef);
      setMetrics({
        rvArea: result.rvCount, myoArea: result.myoCount, lvArea: result.lvCount,
        diagnosis, myoLvRatio: ratio, ejectionFraction: ef,
        lvVolMl: toMl(result.lvCount), myoVolMl: toMl(result.myoCount), rvVolMl: toMl(result.rvCount),
      });
      setStatus('success');
    } catch (err) { console.error(err); setStatus('idle'); }
  };

  // Feature: 3D Point Cloud Generator
  const generate3DModel = async () => {
    if (!session || !allSlices.length || !sliceMeta) return;
    setStatus('computing_ef'); // reusing progress indicator
    setEfProgress(0);
    setThreeDData(null);
    
    const T = 224, TT = T * T;
    const pts: number[] = [];
    const cls: number[] = []; // colors (r,g,b)
    
    const scaleX = pixdim[0] || 1;
    const scaleY = pixdim[1] || 1;
    const scaleZ = pixdim[2] || 1;

    try {
      for (let s = 0; s < allSlices.length; s++) {
        const url = rawB64ToDataUrl(allSlices[s], sliceMeta.cols, sliceMeta.rows);
        const img = new Image();
        await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = rej; img.src = url; });
        const hc = document.createElement('canvas'); hc.width = T; hc.height = T;
        const hctx = hc.getContext('2d', { willReadFrequently: true })!;
        hctx.drawImage(img, 0, 0, T, T);
        const raw = hctx.getImageData(0, 0, T, T).data;
        const float = new Float32Array(TT);
        let sum = 0;
        for (let i = 0; i < TT; i++) {
          const g = (raw[i*4]*0.299 + raw[i*4+1]*0.587 + raw[i*4+2]*0.114) / 255;
          float[i] = g; sum += g;
        }
        const mean = sum / TT; let vv = 0; float.forEach(x => vv += (x - mean) ** 2);
        const std = Math.sqrt(vv / TT) || 1;
        for (let i = 0; i < TT; i++) float[i] = (float[i] - mean) / std;
        
        const tensor = new ort.Tensor('float32', float, [1, 1, T, T]);
        const out = (await session.run({ [session.inputNames[0]]: tensor }))[session.outputNames[0]];
        const outData = out.data as Float32Array;
        
        let isNHWC = false, nc = 4;
        if (out.dims.length === 4) { if (out.dims[3] <= 10) { isNHWC = true; nc = out.dims[3]; } else nc = out.dims[1]; }
        
        const COLORS = [[0,0,0], [59/255, 130/255, 246/255], [16/255, 185/255, 129/255], [239/255, 68/255, 68/255]];
        
        for (let y = 0; y < T; y++) {
          for (let x = 0; x < T; x++) {
             let px = y * T + x;
             let clsIdx = 0, mx = -Infinity;
             for (let c = 0; c < nc; c++) {
               const v = isNHWC ? outData[px*nc+c] : outData[c*TT+px];
               if (v > mx) { mx = v; clsIdx = c; }
             }
             if (clsIdx > 0 && x % 2 === 0 && y % 2 === 0) {
               pts.push((x - T/2) * scaleX, (y - T/2) * scaleY, (s - allSlices.length/2) * scaleZ);
               cls.push(COLORS[clsIdx][0], COLORS[clsIdx][1], COLORS[clsIdx][2]);
             }
          }
        }
        setEfProgress(Math.round(((s + 1) / allSlices.length) * 100));
        await new Promise(r => setTimeout(r, 10)); // yield
      }
      
      setThreeDData({ points: new Float32Array(pts), colors: new Float32Array(cls) });
      setStatus('success');
    } catch (err) { console.error(err); setStatus('idle'); }
  };

  // Feature 3: PDF export
  const exportPDF = async () => {
    const { default: jsPDF } = await import('jspdf');
    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const W = 210, H = 297;

    // Header
    doc.setFillColor(0,0,0); doc.rect(0,0,W,30,'F');
    doc.setTextColor(255,255,255); doc.setFontSize(18); doc.setFont('helvetica','bold');
    doc.text('OPENCARDIAC AI', 15, 13);
    doc.setFontSize(8); doc.setFont('helvetica','normal');
    doc.text('Automated Cardiac MRI Segmentation Report · U-Net ACDC Model', 15, 20);
    doc.text(`Generated: ${new Date().toLocaleString()}  |  Mode: ${fileMode.toUpperCase()}`, 15, 26);

    // Images
    if (selectedImage) {
      doc.setTextColor(0,0,0); doc.setFontSize(9); doc.setFont('helvetica','bold');
      doc.text(`MRI SCAN (WL:${windowLevel} WW:${windowWidth})`, 15, 42);
      doc.addImage(selectedImage, 'PNG', 15, 46, 78, 78);
    }
    if (canvasRef.current) {
      doc.setFont('helvetica','bold'); doc.text('AI SEGMENTATION MASK', 107, 42);
      doc.addImage(canvasRef.current.toDataURL('image/png'), 'PNG', 107, 46, 78, 78);
    }

    // Diagnostics
    if (metrics) {
      const y0 = 135;
      doc.setFillColor(245,245,245); doc.rect(15,y0-5,W-30,80,'F');
      doc.setTextColor(0,0,0); doc.setFontSize(11); doc.setFont('helvetica','bold');
      doc.text('CLINICAL DIAGNOSTIC REPORT', 15, y0+5);
      doc.setFontSize(9); doc.setFont('helvetica','normal');
      const rows: [string,string][] = [
        ['Clinical Impression',    metrics.diagnosis],
        ['MYO / LV Wall Ratio',    metrics.myoLvRatio > 0 ? metrics.myoLvRatio.toFixed(3) : 'N/A'],
        ...(metrics.ejectionFraction !== undefined ? [['Ejection Fraction (EF)', `${metrics.ejectionFraction}%`] as [string,string]] : []),
        ['LV Volume',              `${metrics.lvVolMl} ml  (${metrics.lvArea} px)`],
        ['Myocardium Volume',      `${metrics.myoVolMl} ml  (${metrics.myoArea} px)`],
        ['RV Volume',              `${metrics.rvVolMl} ml  (${metrics.rvArea} px)`],
        ['Voxel Spacing',          pixdim.map(p => p.toFixed(2)).join(' × ') + ' mm'],
        ['Slices Averaged',        allSlices.length > 1 ? `Multi-slice (3-slice avg)` : 'Single slice'],
      ];
      rows.forEach(([l,v], i) => {
        doc.setFont('helvetica','bold'); doc.text(l+':', 20, y0+17+i*8);
        doc.setFont('helvetica','normal'); doc.text(v, 100, y0+17+i*8);
      });
    }

    // Wall thickness summary
    if (wallThicknesses.some(t => t > 0)) {
      const vox = (pixdim[0]+pixdim[1])/2;
      doc.setFont('helvetica','bold'); doc.setFontSize(9);
      doc.text('WALL THICKNESS (12 SECTORS):', 15, 235);
      doc.setFont('helvetica','normal'); doc.setFontSize(8);
      const thickStr = wallThicknesses.map((t,i) => `S${i+1}: ${(t*vox).toFixed(1)}mm`).join('  ');
      doc.text(thickStr, 15, 242);
    }

    // Legend + footer
    [[239,68,68,'LV'],[16,185,129,'MYO'],[59,130,246,'RV']].forEach(([r,g,b,l],i) => {
      doc.setFillColor(r as number,g as number,b as number);
      doc.rect(15,248+i*7,6,4,'F');
      doc.setFont('helvetica','normal'); doc.setFontSize(8);
      doc.setTextColor(0,0,0); doc.text(l as string, 24, 251+i*7);
    });
    doc.setFillColor(0,0,0); doc.rect(0,H-12,W,12,'F');
    doc.setTextColor(130,130,130); doc.setFontSize(6);
    doc.text('OpenCardiac AI · Research Use Only — Not a Substitute for Clinical Diagnosis', 15, H-5);
    doc.save('opencardiac_report.pdf');
  };

  return (
    <div className="app-shell">
      {/* ── HEADER ── */}
      <header className="header">
        <div className="brand">
          <div className="brand-icon"><Heart size={18} /></div>
          <div className="brand-text">
            <h1>OpenCardiac AI</h1>
            <p>Advanced Cardiac MRI Analysis Platform</p>
          </div>
        </div>
        <div className={`header-status ${engineReady ? 'ready' : ''}`}>
          {engineReady && <div className="status-dot" />}
          {engineReady ? 'Engine Ready' : 'Loading Model...'}
        </div>
      </header>

      <div className="workspace">
        {/* ── LEFT SIDEBAR ── */}
        <aside className="sidebar">

          {/* Upload */}
          <div className="sidebar-section fade-in">
            <h2 className="section-title"><LayoutDashboard size={14} /> Control Panel</h2>
            <input type="file" accept="image/*,.dcm,.nii,.nii.gz" ref={fileInputRef}
              style={{ display: 'none' }} onChange={e => e.target.files?.[0] && loadFile(e.target.files[0])} />
            <button className="btn btn-upload" onClick={() => fileInputRef.current?.click()}>
              <ImageUp size={16} /> Load Scan
            </button>
            <div style={{ marginTop: '0.4rem', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              PNG · JPG · DCM · NIfTI (.nii.gz)
            </div>
          </div>

          {/* Feature 3: Windowing */}
          {allSlices.length > 0 && (
            <div className="sidebar-section fade-in">
              <h2 className="section-title"><SlidersHorizontal size={14} /> Windowing (WW/WL)</h2>
              <div className="info-row" style={{ marginBottom: '0.3rem' }}>
                <span className="info-label">Level (WL)</span>
                <span className="info-value">{windowLevel}</span>
              </div>
              <input type="range" min={0} max={255} value={windowLevel}
                onChange={e => setWindowLevel(Number(e.target.value))} className="slice-slider" />
              <div className="info-row" style={{ marginTop: '0.7rem', marginBottom: '0.3rem' }}>
                <span className="info-label">Width (WW)</span>
                <span className="info-value">{windowWidth}</span>
              </div>
              <input type="range" min={10} max={255} value={windowWidth}
                onChange={e => setWindowWidth(Number(e.target.value))} className="slice-slider" />
            </div>
          )}

          {/* Analysis controls */}
          {selectedImage && (
            <div className="sidebar-section fade-in">
              <h2 className="section-title"><Zap size={14} /> Analysis</h2>

              {/* Z-scrubber */}
              {sliceMeta && sliceMeta.sliceCount > 1 && (
                <div style={{ marginBottom: '1rem' }}>
                  <div className="info-row" style={{ marginBottom: '0.4rem' }}>
                    <span className="info-label">{fileMode === '4d_cine' ? 'Time Frame' : 'Z-Slice'}</span>
                    <span className="info-value">{currentSlice + 1} / {sliceMeta.sliceCount}</span>
                  </div>
                  <input type="range" min={0} max={sliceMeta.sliceCount - 1} value={currentSlice}
                    onChange={e => onSliceChange(Number(e.target.value))} className="slice-slider" />
                </div>
              )}

              <button className="btn btn-run" onClick={runSegment}
                disabled={status === 'processing' || status === 'computing_ef' || !engineReady}>
                <Play size={14} fill="currentColor" />
                {status === 'processing' ? 'Segmenting...' : 'Segment Slice'}
              </button>

              {fileMode === '4d_cine' && (
                <button className="btn btn-upload" style={{ marginTop: '0.5rem' }} onClick={runEF}
                  disabled={status === 'processing' || status === 'computing_ef' || !engineReady}>
                  <FileHeart size={14} />
                  {status === 'computing_ef' ? `EF: ${efProgress}%` : 'Compute Ejection Fraction'}
                </button>
              )}

              {fileMode === '3d_volume' && allSlices.length > 1 && (
                <button className="btn btn-upload" style={{ marginTop: '0.5rem' }} onClick={generate3DModel}
                  disabled={status === 'processing' || status === 'computing_ef' || !engineReady}>
                  <Layers size={14} />
                  {status === 'computing_ef' ? `Building 3D: ${efProgress}%` : 'Generate 3D Heart (Points)'}
                </button>
              )}

              {/* Feature 1: Confidence toggle */}
              {maskDataUrl && confidenceMapUrl && (
                <button className="btn btn-clear" style={{ marginTop: '0.5rem' }}
                  onClick={() => {
                    const next = !showConfidence;
                    setShowConfidence(next);
                    drawOnCanvas(next ? confidenceMapUrl : maskDataUrl);
                  }}>
                  {showConfidence ? <EyeOff size={13} /> : <Eye size={13} />}
                  {showConfidence ? 'Show RGB Mask' : 'Show Confidence Map'}
                </button>
              )}

              <button className="btn btn-clear" style={{ marginTop: '0.5rem' }} onClick={resetState}>
                <Trash2 size={13} /> Reset
              </button>
            </div>
          )}

          {/* Diagnostics */}
          <div className="sidebar-section fade-in-2">
            <h2 className="section-title"><Activity size={14} /> Diagnostics</h2>
            <div className="info-list">
              <div className="info-row">
                <span className="info-label">Status</span>
                <span className="info-value">
                  {status === 'idle' && 'Awaiting Input'}
                  {status === 'processing' && 'Analyzing...'}
                  {status === 'computing_ef' && `EF ${efProgress}%`}
                  {status === 'success' && 'Complete'}
                </span>
              </div>
              {metrics && (
                <div style={{ borderTop: '1px solid var(--border-subtle)', marginTop: '0.5rem', paddingTop: '0.75rem' }}>
                  {/* Feature 4: Volumes in ml */}
                  <div className="info-row">
                    <span className="info-label">LV Volume</span>
                    <span className="info-value">{metrics.lvVolMl} ml</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">MYO Volume</span>
                    <span className="info-value">{metrics.myoVolMl} ml</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">RV Volume</span>
                    <span className="info-value">{metrics.rvVolMl} ml</span>
                  </div>
                  <div className="info-row" style={{ marginTop: '0.4rem' }}>
                    <span className="info-label">MYO / LV</span>
                    <span className="info-value" style={{ color: (metrics.myoLvRatio > 1.8 || (metrics.myoLvRatio < 0.6 && metrics.myoLvRatio > 0)) ? '#ef4444' : '#10b981' }}>
                      {metrics.myoLvRatio > 0 ? metrics.myoLvRatio.toFixed(2) : 'N/A'}
                    </span>
                  </div>
                  {metrics.ejectionFraction !== undefined && (
                    <div className="info-row">
                      <span className="info-label">Ejection Fraction</span>
                      <span className="info-value" style={{ color: metrics.ejectionFraction < 50 ? '#ef4444' : '#10b981' }}>
                        {metrics.ejectionFraction}%
                      </span>
                    </div>
                  )}
                  <div className="info-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.25rem', marginTop: '0.4rem' }}>
                    <span className="info-label">Clinical Impression</span>
                    <span className="info-value" style={{ color: metrics.diagnosis === "Normal Cardiac Structure" ? '#10b981' : '#ef4444', lineHeight: 1.4, fontSize: '0.8rem' }}>
                      {metrics.diagnosis}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Mask Legend */}
          <div className="sidebar-section fade-in-2">
            <h2 className="section-title"><Layers size={14} /> {showConfidence ? 'Confidence Scale' : 'Mask Legend'}</h2>
            {showConfidence ? (
              <div style={{ fontSize: '0.7rem', lineHeight: 2 }}>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <div style={{ width: 30, height: 8, background: 'linear-gradient(to right, #0000ff, #00c800, #ff0000)' }} />
                  <span style={{ color: 'var(--text-muted)' }}>Low → High confidence</span>
                </div>
              </div>
            ) : (
              [['#ef4444','Left Ventricle (LV)'],['#10b981','Myocardium (MYO)'],['#3b82f6','Right Ventricle (RV)']].map(([c,l]) => (
                <div key={l} className="legend-item">
                  <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                    <div className="legend-swatch" style={{ background: c, borderColor: c }} />
                    <span className="info-label">{l}</span>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* PDF Export */}
          {status === 'success' && metrics && (
            <div className="sidebar-section fade-in">
              <h2 className="section-title"><Download size={14} /> Export</h2>
              <button className="btn btn-run" onClick={exportPDF}>
                <Download size={14} /> Export PDF Report
              </button>
            </div>
          )}

          {/* Engine specs */}
          <div className="sidebar-section fade-in-3" style={{ flex: 1 }}>
            <h2 className="section-title"><Cpu size={14} /> Engine</h2>
            <div className="info-list">
              <div className="info-row"><span className="info-label">Backend</span><span className="info-value">WASM</span></div>
              <div className="info-row"><span className="info-label">Input</span><span className="info-value">1×1×224×224</span></div>
              <div className="info-row"><span className="info-label">Classes</span><span className="info-value">BG/RV/MYO/LV</span></div>
              {allSlices.length > 1 && <div className="info-row"><span className="info-label">Inference</span><span className="info-value">3-Slice Avg</span></div>}
              {sliceMeta && <div className="info-row"><span className="info-label">Voxel</span><span className="info-value">{pixdim.map(p=>p.toFixed(1)).join('×')}mm</span></div>}
            </div>
          </div>
        </aside>

        {/* ── MAIN STAGE ── */}
        <main className="main-stage">
          <div className="viewer-container">
            {!selectedImage ? (
              <div className="viewer-empty fade-in-2" onClick={() => fileInputRef.current?.click()} style={{ cursor: 'pointer' }}>
                <ImageUp size={48} />
                <p style={{ textTransform: 'uppercase', letterSpacing: '0.05em', fontSize: '0.85rem' }}>Click to Load Scan</p>
                <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>PNG · JPG · DCM · NIfTI (.nii.gz)</p>
              </div>
            ) : (
              <>
                <img id="mri-img" className="viewer-img" src={selectedImage} alt="MRI" />
                <canvas ref={canvasRef} className="viewer-canvas" />
                {status === 'success' && (
                  <div className="overlay-status fade-in">
                    {showConfidence ? '🔵 Confidence Map' : '✓ Mask Active'}
                  </div>
                )}
                {(status === 'processing' || status === 'computing_ef') && (
                  <>
                    <div className="overlay-processing">
                      <div className="spinner" />
                      <p style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                        {status === 'computing_ef'
                          ? `Computing EF — ${efProgress}% (${allSlices.length} frames)`
                          : allSlices.length > 1 ? '3-Slice Multi-Slice Inference...' : 'Segmenting structures...'}
                      </p>
                    </div>
                    <div className="scanline" />
                  </>
                )}
              </>
            )}
          </div>

          {sliceMeta && (
            <div style={{ marginTop: '0.75rem', fontSize: '0.72rem', color: 'var(--text-muted)', display: 'flex', gap: '1.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              <span>Mode: <strong style={{ color: 'var(--text-secondary)' }}>{fileMode.replace('_',' ')}</strong></span>
              <span>Dims: <strong style={{ color: 'var(--text-secondary)' }}>{sliceMeta.cols}×{sliceMeta.rows}</strong></span>
              <span>Frames: <strong style={{ color: 'var(--text-secondary)' }}>{sliceMeta.sliceCount}</strong></span>
              <span>Voxel: <strong style={{ color: 'var(--text-secondary)' }}>{pixdim.map(p=>p.toFixed(1)).join('×')}mm</strong></span>
            </div>
          )}

          {/* Feature 5: Cardiac Phase Chart */}
          {lvAreaCurve.length > 1 && (
            <div style={{ marginTop: '1.5rem', width: '100%', maxWidth: 700 }}>
              <h3 style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <TrendingUp size={13} /> Feature 5 — Cardiac Phase: LV Area Curve ({allSlices.length} frames)
              </h3>
              <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', padding: '1rem' }}>
                <PhaseChart lvAreas={lvAreaCurve} dIdx={diastoleIdx} sIdx={systoleIdx} />
              </div>
            </div>
          )}

          {/* Feature 6: Wall Thickness Polar Chart */}
          {wallThicknesses.length > 0 && wallThicknesses.some(t => t > 0) && (
            <div style={{ marginTop: '1.5rem', width: '100%', maxWidth: 700 }}>
              <h3 style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                ⬡ Feature 6 — Myocardial Wall Thickness (12-Sector Polar Map)
              </h3>
              <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', padding: '1.25rem' }}>
                <WallThicknessChart thicknesses={wallThicknesses} pixdim={pixdim} />
              </div>
            </div>
          )}

          {/* 3D Viewer */}
          {threeDData && (
            <div style={{ marginTop: '1.5rem', width: '100%', maxWidth: 700 }}>
              <h3 style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                <Layers size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }}/>
                Interactive 3D Point Cloud Volume
              </h3>
              <ThreeDViewer points={threeDData.points} colors={threeDData.colors} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
