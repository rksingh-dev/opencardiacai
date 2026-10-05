"use client";

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Activity, Heart, Layers, ImageUp, Play, Trash2,
  Cpu, LayoutDashboard, Download, Zap, FileHeart
} from 'lucide-react';
import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

type Status = "idle" | "processing" | "success" | "computing_ef";
type FileMode = 'image' | 'dicom' | '3d_volume' | '4d_cine';

interface Diagnostics {
  rvArea: number;
  myoArea: number;
  lvArea: number;
  diagnosis: string;
  myoLvRatio: number;
  ejectionFraction?: number;
}

// Decode base64 pixel bytes into a grayscale PNG dataURL
function sliceToDataURL(b64: string, cols: number, rows: number): string {
  const binaryString = atob(b64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);

  const canvas = document.createElement('canvas');
  canvas.width = cols; canvas.height = rows;
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.createImageData(cols, rows);
  for (let i = 0; i < bytes.length; i++) {
    const idx = i * 4;
    imgData.data[idx] = bytes[i];
    imgData.data[idx + 1] = bytes[i];
    imgData.data[idx + 2] = bytes[i];
    imgData.data[idx + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}

// Run segmentation on a single dataURL, return mask pixel counts
async function runSegmentation(
  dataUrl: string,
  session: ort.InferenceSession
): Promise<{ rvCount: number; myoCount: number; lvCount: number; maskDataUrl: string }> {
  const targetSize = 224;
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = reject;
    img.src = dataUrl;
  });

  const hc = document.createElement('canvas');
  hc.width = targetSize; hc.height = targetSize;
  const hctx = hc.getContext('2d', { willReadFrequently: true })!;
  hctx.drawImage(img, 0, 0, targetSize, targetSize);

  const raw = hctx.getImageData(0, 0, targetSize, targetSize).data;
  const float = new Float32Array(targetSize * targetSize);
  let sum = 0;
  for (let i = 0; i < targetSize * targetSize; i++) {
    const g = (raw[i * 4] * 0.299 + raw[i * 4 + 1] * 0.587 + raw[i * 4 + 2] * 0.114) / 255;
    float[i] = g; sum += g;
  }
  const mean = sum / float.length;
  let variance = 0;
  float.forEach(v => variance += (v - mean) ** 2);
  const std = Math.sqrt(variance / float.length) || 1;
  for (let i = 0; i < float.length; i++) float[i] = (float[i] - mean) / std;

  const tensor = new ort.Tensor('float32', float, [1, 1, targetSize, targetSize]);
  const results = await session.run({ [session.inputNames[0]]: tensor });
  const out = results[session.outputNames[0]];
  const outData = out.data as Float32Array;

  let isNHWC = false;
  let numClasses = 4;
  if (out.dims.length === 4) {
    if (out.dims[3] <= 10) { isNHWC = true; numClasses = out.dims[3]; }
    else { numClasses = out.dims[1]; }
  }

  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = targetSize; maskCanvas.height = targetSize;
  const mctx = maskCanvas.getContext('2d')!;
  const maskImgData = mctx.createImageData(targetSize, targetSize);
  const colors = [
    [0, 0, 0, 0],
    [59, 130, 246, 160],  // RV Blue
    [16, 185, 129, 160],  // MYO Green
    [239, 68, 68, 160],   // LV Red
  ];

  let rvCount = 0, myoCount = 0, lvCount = 0;
  for (let y = 0; y < targetSize; y++) {
    for (let x = 0; x < targetSize; x++) {
      const px = y * targetSize + x;
      let maxClass = 0;
      let maxVal = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        const v = isNHWC ? outData[px * numClasses + c] : outData[c * (targetSize * targetSize) + px];
        if (v > maxVal) { maxVal = v; maxClass = c; }
      }
      if (maxClass === 1) rvCount++;
      if (maxClass === 2) myoCount++;
      if (maxClass === 3) lvCount++;
      const [r, g, b, a] = colors[maxClass] ?? [0, 0, 0, 0];
      const i4 = px * 4;
      maskImgData.data[i4] = r; maskImgData.data[i4 + 1] = g;
      maskImgData.data[i4 + 2] = b; maskImgData.data[i4 + 3] = a;
    }
  }
  mctx.putImageData(maskImgData, 0, 0);
  return { rvCount, myoCount, lvCount, maskDataUrl: maskCanvas.toDataURL('image/png') };
}

function getDiagnosis(myoCount: number, lvCount: number, rvCount: number, ef?: number): { diagnosis: string; ratio: number } {
  let diagnosis = "Normal Cardiac Structure";
  let ratio = 0;
  if (lvCount < 50 || myoCount < 50) {
    diagnosis = "Inconclusive (Incomplete View)";
  } else {
    ratio = myoCount / lvCount;
    if (ef !== undefined && ef < 35) diagnosis = "Severe Systolic Dysfunction";
    else if (ef !== undefined && ef < 50) diagnosis = "Possible Dilated Cardiomyopathy (DCM)";
    else if (ratio > 1.8) diagnosis = "Possible Hypertrophic Cardiomyopathy (HCM)";
    else if (ratio < 0.6) diagnosis = "Possible Dilated Cardiomyopathy (DCM)";
    else if (rvCount > lvCount * 2.5) diagnosis = "Abnormal Right Ventricle (ARV)";
  }
  return { diagnosis, ratio };
}

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [maskDataUrl, setMaskDataUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [engineReady, setEngineReady] = useState(false);
  const [metrics, setMetrics] = useState<Diagnostics | null>(null);

  // Z-Axis Scrubber state
  const [allSlices, setAllSlices] = useState<string[]>([]);
  const [sliceMeta, setSliceMeta] = useState<{ cols: number; rows: number; sliceCount: number } | null>(null);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [fileMode, setFileMode] = useState<FileMode>('image');

  // EF state
  const [efProgress, setEfProgress] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    async function loadModel() {
      try {
        const s = await ort.InferenceSession.create('/cine_model.onnx', { executionProviders: ['wasm'] });
        setSession(s);
        setEngineReady(true);
      } catch (err) { console.error("Model load error:", err); }
    }
    loadModel();
  }, []);

  const resetState = () => {
    setSelectedImage(null);
    setMaskDataUrl(null);
    setStatus("idle");
    setMetrics(null);
    setAllSlices([]);
    setSliceMeta(null);
    setCurrentSlice(0);
    setFileMode('image');
    setEfProgress(0);
  };

  const renderMaskOnCanvas = (maskUrl: string) => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    canvas.width = 224; canvas.height = 224;
    const maskImg = new Image();
    maskImg.onload = () => ctx.drawImage(maskImg, 0, 0);
    maskImg.src = maskUrl;
  };

  const loadFile = async (file: File) => {
    resetState();
    const fname = file.name.toLowerCase();

    if (fname.endsWith('.dcm') || fname.endsWith('.nii') || fname.endsWith('.nii.gz')) {
      setStatus('processing');
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch('/api/dicom', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);

        const { cols, rows, sliceCount, mode, slices } = data;
        setSliceMeta({ cols, rows, sliceCount });
        setFileMode(mode as FileMode);
        setAllSlices(slices);

        // Show the middle slice initially
        const midIdx = Math.floor(sliceCount / 2);
        setCurrentSlice(midIdx);
        setSelectedImage(sliceToDataURL(slices[midIdx], cols, rows));
        setStatus('idle');
      } catch (err) {
        console.error(err);
        alert("Failed to parse medical image.");
        setStatus('idle');
      }
      return;
    }

    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string);
      setFileMode('image');
      setStatus("idle");
    };
    reader.readAsDataURL(file);
  };

  // Z-scrubber: update the displayed image when slider changes
  const onSliceChange = (idx: number) => {
    if (!sliceMeta || allSlices.length === 0) return;
    setCurrentSlice(idx);
    setSelectedImage(sliceToDataURL(allSlices[idx], sliceMeta.cols, sliceMeta.rows));
    // Clear old mask
    setMaskDataUrl(null);
    setMetrics(null);
    setStatus('idle');
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')!;
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  // Feature 2: Standard single-slice segmentation
  const runSegmentationSingle = async () => {
    if (!selectedImage || !session) return;
    setStatus('processing');
    try {
      const { rvCount, myoCount, lvCount, maskDataUrl: mUrl } = await runSegmentation(selectedImage, session);
      setMaskDataUrl(mUrl);
      renderMaskOnCanvas(mUrl);
      const { diagnosis, ratio } = getDiagnosis(myoCount, lvCount, rvCount);
      setMetrics({ rvArea: rvCount, myoArea: myoCount, lvArea: lvCount, myoLvRatio: ratio, diagnosis });
      setStatus('success');
    } catch (err) { console.error(err); setStatus('idle'); }
  };

  // Feature 1: EF calculation by running model on all time frames (4D cine)
  const runEjectionFraction = async () => {
    if (!session || allSlices.length === 0 || !sliceMeta) return;
    setStatus('computing_ef');
    setEfProgress(0);

    try {
      let maxLv = -Infinity, minLv = Infinity;
      let bestDiastoleIdx = 0, bestSystoleIdx = 0;

      for (let i = 0; i < allSlices.length; i++) {
        const frameUrl = sliceToDataURL(allSlices[i], sliceMeta.cols, sliceMeta.rows);
        const { lvCount } = await runSegmentation(frameUrl, session);
        if (lvCount > maxLv) { maxLv = lvCount; bestDiastoleIdx = i; }
        if (lvCount < minLv) { minLv = lvCount; bestSystoleIdx = i; }
        setEfProgress(Math.round(((i + 1) / allSlices.length) * 100));
      }

      // EF = (EDV - ESV) / EDV × 100 (simplified using pixel area as volume proxy)
      const ef = maxLv > 0 ? Math.round(((maxLv - minLv) / maxLv) * 100) : 0;

      // Show diastole frame with full segmentation
      const diastoleUrl = sliceToDataURL(allSlices[bestDiastoleIdx], sliceMeta.cols, sliceMeta.rows);
      const { rvCount, myoCount, lvCount, maskDataUrl: mUrl } = await runSegmentation(diastoleUrl, session);
      setSelectedImage(diastoleUrl);
      setCurrentSlice(bestDiastoleIdx);
      setMaskDataUrl(mUrl);
      renderMaskOnCanvas(mUrl);

      const { diagnosis, ratio } = getDiagnosis(myoCount, lvCount, rvCount, ef);
      setMetrics({ rvArea: rvCount, myoArea: myoCount, lvArea: lvCount, myoLvRatio: ratio, diagnosis, ejectionFraction: ef });
      setStatus('success');
    } catch (err) { console.error(err); setStatus('idle'); }
  };

  // Feature 3: PDF Export
  const exportPDF = async () => {
    const { default: jsPDF } = await import('jspdf');
    const { default: html2canvas } = await import('html2canvas');

    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const pageW = 210, pageH = 297;

    // Header
    doc.setFillColor(0, 0, 0);
    doc.rect(0, 0, pageW, 30, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20);
    doc.setFont('helvetica', 'bold');
    doc.text('OPENCARDIAC AI', 15, 13);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    doc.text('Automated Cardiac MRI Segmentation Report', 15, 20);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 15, 26);

    // MRI Image
    if (selectedImage) {
      doc.setTextColor(0, 0, 0);
      doc.setFontSize(11);
      doc.setFont('helvetica', 'bold');
      doc.text('MRI SCAN', 15, 42);
      doc.addImage(selectedImage, 'PNG', 15, 46, 80, 80);
    }

    // Mask Overlay
    if (canvasRef.current) {
      const maskImg = canvasRef.current.toDataURL('image/png');
      doc.setFontSize(11);
      doc.setFont('helvetica', 'bold');
      doc.text('AI SEGMENTATION MASK', 110, 42);
      doc.addImage(maskImg, 'PNG', 110, 46, 80, 80);
    }

    // Diagnostics
    if (metrics) {
      const yStart = 140;
      doc.setFillColor(245, 245, 245);
      doc.rect(15, yStart - 5, pageW - 30, 80, 'F');
      doc.setTextColor(0, 0, 0);
      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.text('CLINICAL DIAGNOSTIC REPORT', 15, yStart + 5);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      const items: [string, string][] = [
        ['Clinical Impression', metrics.diagnosis],
        ['MYO / LV Wall Ratio', metrics.myoLvRatio > 0 ? metrics.myoLvRatio.toFixed(3) : 'N/A'],
        ['LV Area (pixels)', `${metrics.lvArea}`],
        ['Myocardium Area (pixels)', `${metrics.myoArea}`],
        ['RV Area (pixels)', `${metrics.rvArea}`],
        ...(metrics.ejectionFraction !== undefined
          ? [['Ejection Fraction (EF)', `${metrics.ejectionFraction}%`] as [string, string]]
          : []),
      ];
      items.forEach(([label, value], i) => {
        doc.setFont('helvetica', 'bold');
        doc.text(label + ':', 20, yStart + 18 + i * 10);
        doc.setFont('helvetica', 'normal');
        doc.text(value, 100, yStart + 18 + i * 10);
      });
    }

    // Legend
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.text('SEGMENTATION LEGEND', 15, 235);
    doc.setFont('helvetica', 'normal');
    const legend: [number, number, number, string][] = [
      [239, 68, 68, 'Left Ventricle (LV) — Red'],
      [16, 185, 129, 'Myocardium (MYO) — Green'],
      [59, 130, 246, 'Right Ventricle (RV) — Blue'],
    ];
    legend.forEach(([r, g, b, label], i) => {
      doc.setFillColor(r, g, b);
      doc.rect(15, 240 + i * 8, 8, 5, 'F');
      doc.text(label, 26, 244 + i * 8);
    });

    // Footer
    doc.setFillColor(0, 0, 0);
    doc.rect(0, pageH - 15, pageW, 15, 'F');
    doc.setTextColor(150, 150, 150);
    doc.setFontSize(7);
    doc.text('OpenCardiac AI · U-Net ACDC Model · For Research Purposes Only — Not for Clinical Diagnosis', 15, pageH - 6);

    doc.save('opencardiac_report.pdf');
  };

  return (
    <div className="app-shell">
      {/* HEADER */}
      <header className="header">
        <div className="brand">
          <div className="brand-icon"><Heart size={18} /></div>
          <div className="brand-text">
            <h1>OpenCardiac AI</h1>
            <p>Cardiac MRI Analysis Platform</p>
          </div>
        </div>
        <div className={`header-status ${engineReady ? 'ready' : ''}`}>
          {engineReady && <div className="status-dot" />}
          {engineReady ? 'Engine Ready' : 'Loading Model...'}
        </div>
      </header>

      <div className="workspace">
        {/* SIDEBAR */}
        <aside className="sidebar">
          {/* Upload */}
          <div className="sidebar-section fade-in">
            <h2 className="section-title"><LayoutDashboard size={14} /> Control Panel</h2>
            <input
              type="file"
              accept="image/*,.dcm,.nii,.nii.gz"
              ref={fileInputRef}
              onChange={e => e.target.files?.[0] && loadFile(e.target.files[0])}
              style={{ display: 'none' }}
            />
            <button className="btn btn-upload" onClick={() => fileInputRef.current?.click()}>
              <ImageUp size={16} /> Load Scan
            </button>
            <div style={{ marginTop: '0.5rem', fontSize: '0.65rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Supports PNG · JPG · DCM · NIfTI
            </div>
          </div>

          {/* Analysis Controls */}
          {selectedImage && (
            <div className="sidebar-section fade-in">
              <h2 className="section-title"><Zap size={14} /> Analysis</h2>

              {/* Z-Axis Scrubber — Feature 2 */}
              {sliceMeta && sliceMeta.sliceCount > 1 && (
                <div style={{ marginBottom: '1rem' }}>
                  <div className="info-row" style={{ marginBottom: '0.4rem' }}>
                    <span className="info-label">
                      {fileMode === '4d_cine' ? 'Time Frame' : 'Z-Slice'}
                    </span>
                    <span className="info-value">{currentSlice + 1} / {sliceMeta.sliceCount}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={sliceMeta.sliceCount - 1}
                    value={currentSlice}
                    onChange={e => onSliceChange(Number(e.target.value))}
                    className="slice-slider"
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                    <span>0</span>
                    <span>{sliceMeta.sliceCount - 1}</span>
                  </div>
                </div>
              )}

              <button
                className="btn btn-run"
                onClick={runSegmentationSingle}
                disabled={status === 'processing' || status === 'computing_ef' || !engineReady}
              >
                <Play size={14} fill="currentColor" />
                {status === 'processing' ? 'Analyzing...' : 'Segment Slice'}
              </button>

              {/* EF Button — Feature 1 — only for 4D cine */}
              {fileMode === '4d_cine' && (
                <button
                  className="btn btn-upload"
                  style={{ marginTop: '0.5rem' }}
                  onClick={runEjectionFraction}
                  disabled={status === 'processing' || status === 'computing_ef' || !engineReady}
                >
                  <FileHeart size={14} />
                  {status === 'computing_ef' ? `Computing EF... ${efProgress}%` : 'Compute Ejection Fraction'}
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
                  {status === 'computing_ef' && `EF Calc (${efProgress}%)`}
                  {status === 'success' && 'Complete'}
                </span>
              </div>
              {metrics && (
                <>
                  <div className="info-row" style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem', marginTop: '0.5rem' }}>
                    <span className="info-label">LV Area</span>
                    <span className="info-value">{metrics.lvArea}px</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">MYO Area</span>
                    <span className="info-value">{metrics.myoArea}px</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">RV Area</span>
                    <span className="info-value">{metrics.rvArea}px</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">MYO / LV</span>
                    <span className="info-value" style={{ color: (metrics.myoLvRatio > 1.8 || metrics.myoLvRatio < 0.6) ? '#ef4444' : '#10b981' }}>
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
                  <div className="info-row" style={{ flexDirection: 'column', gap: '0.25rem', alignItems: 'flex-start', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem', marginTop: '0.25rem' }}>
                    <span className="info-label">Clinical Impression</span>
                    <span className="info-value" style={{ color: metrics.diagnosis === "Normal Cardiac Structure" ? '#10b981' : '#ef4444', lineHeight: 1.4 }}>
                      {metrics.diagnosis}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Legend */}
          <div className="sidebar-section fade-in-2">
            <h2 className="section-title"><Layers size={14} /> Mask Legend</h2>
            <div className="legend-item">
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                <div className="legend-swatch" style={{ background: '#ef4444', borderColor: '#ef4444' }} />
                <span className="info-label">Left Ventricle (LV)</span>
              </div>
            </div>
            <div className="legend-item">
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                <div className="legend-swatch" style={{ background: '#10b981', borderColor: '#10b981' }} />
                <span className="info-label">Myocardium (MYO)</span>
              </div>
            </div>
            <div className="legend-item">
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                <div className="legend-swatch" style={{ background: '#3b82f6', borderColor: '#3b82f6' }} />
                <span className="info-label">Right Ventricle (RV)</span>
              </div>
            </div>
          </div>

          {/* PDF Export — Feature 3 */}
          {status === 'success' && metrics && (
            <div className="sidebar-section fade-in">
              <h2 className="section-title"><Download size={14} /> Export</h2>
              <button className="btn btn-run" onClick={exportPDF}>
                <Download size={14} /> Export PDF Report
              </button>
            </div>
          )}

          {/* Engine Specs */}
          <div className="sidebar-section fade-in-3" style={{ flex: 1 }}>
            <h2 className="section-title"><Cpu size={14} /> Engine</h2>
            <div className="info-list">
              <div className="info-row">
                <span className="info-label">Backend</span>
                <span className="info-value">WASM</span>
              </div>
              <div className="info-row">
                <span className="info-label">Input</span>
                <span className="info-value">1×1×224×224</span>
              </div>
              <div className="info-row">
                <span className="info-label">Classes</span>
                <span className="info-value">4 (BG,RV,MYO,LV)</span>
              </div>
            </div>
          </div>
        </aside>

        {/* MAIN VIEWER */}
        <main className="main-stage">
          <div className="viewer-container" ref={viewerRef}>
            {!selectedImage ? (
              <div
                className="viewer-empty fade-in-2"
                onClick={() => fileInputRef.current?.click()}
                style={{ cursor: 'pointer' }}
              >
                <ImageUp size={48} />
                <p style={{ textTransform: 'uppercase', letterSpacing: '0.05em', fontSize: '0.85rem' }}>Click to Load Scan</p>
                <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>PNG · JPG · DCM · NIfTI (.nii.gz)</p>
              </div>
            ) : (
              <>
                <img id="mri-img" className="viewer-img" src={selectedImage} alt="MRI" />
                <canvas ref={canvasRef} className="viewer-canvas" />

                {status === 'success' && (
                  <div className="overlay-status fade-in">Mask Active</div>
                )}

                {(status === 'processing' || status === 'computing_ef') && (
                  <>
                    <div className="overlay-processing">
                      <div className="spinner" />
                      <p style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                        {status === 'computing_ef'
                          ? `Computing EF — ${efProgress}% (${allSlices.length} frames)`
                          : 'Segmenting structures...'}
                      </p>
                    </div>
                    <div className="scanline" />
                  </>
                )}
              </>
            )}
          </div>

          {/* Slice mode info bar */}
          {sliceMeta && (
            <div style={{
              marginTop: '0.75rem', fontSize: '0.72rem', color: 'var(--text-muted)',
              display: 'flex', gap: '1.5rem', textTransform: 'uppercase', letterSpacing: '0.05em'
            }}>
              <span>Mode: <strong style={{ color: 'var(--text-secondary)' }}>{fileMode.replace('_', ' ').toUpperCase()}</strong></span>
              <span>Dims: <strong style={{ color: 'var(--text-secondary)' }}>{sliceMeta.cols}×{sliceMeta.rows}</strong></span>
              <span>Frames: <strong style={{ color: 'var(--text-secondary)' }}>{sliceMeta.sliceCount}</strong></span>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
