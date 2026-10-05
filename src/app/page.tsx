"use client";

import React, { useState, useRef, useEffect } from 'react';
import {
  Activity, Heart, Layers, ImageUp, Play, Trash2,
  Cpu, Zap, CheckCircle2, FileStack
} from 'lucide-react';
import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

type Status = "idle" | "processing" | "success";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [engineReady, setEngineReady] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    async function loadModel() {
      try {
        const s = await ort.InferenceSession.create('/cine_model.onnx', {
          executionProviders: ['wasm']
        });
        setSession(s);
        setEngineReady(true);
      } catch (err) {
        console.error("Model load error:", err);
      }
    }
    loadModel();
  }, []);

  const loadFile = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string);
      setStatus("idle");
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  };

  const runPrediction = async () => {
    if (!selectedImage || !session) return;
    setStatus("processing");

    try {
      const img = document.getElementById("mri-img") as HTMLImageElement;
      const targetSize = 224;
      const hc = hiddenCanvasRef.current!;
      const hctx = hc.getContext('2d', { willReadFrequently: true })!;
      hc.width = targetSize; hc.height = targetSize;
      hctx.drawImage(img, 0, 0, targetSize, targetSize);

      const raw = hctx.getImageData(0, 0, targetSize, targetSize).data;
      const float = new Float32Array(targetSize * targetSize);
      let sum = 0;

      for (let i = 0; i < targetSize * targetSize; i++) {
        const g = (raw[i*4]*0.299 + raw[i*4+1]*0.587 + raw[i*4+2]*0.114) / 255;
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
      } else if (out.dims.length === 3) { isNHWC = true; numClasses = 1; }

      const canvas = canvasRef.current!;
      const ctx = canvas.getContext('2d')!;
      canvas.width = targetSize; canvas.height = targetSize;
      const imgData = ctx.createImageData(targetSize, targetSize);

      // Monochrome palette: RV=dark gray, MYO=light gray, LV=pure white
      const colors = [
        [0,0,0,0],        // Background (Transparent)
        [255,255,255,80], // RV (Subtle white)
        [255,255,255,140],// MYO (Medium white)
        [255,255,255,220] // LV (Bright white)
      ];

      for (let y = 0; y < targetSize; y++) {
        for (let x = 0; x < targetSize; x++) {
          const px = y * targetSize + x;
          let maxClass = 0;
          if (numClasses === 1) {
            maxClass = Math.round(outData[px]);
          } else {
            let maxVal = -Infinity;
            for (let c = 0; c < numClasses; c++) {
              const v = isNHWC ? outData[px*numClasses+c] : outData[c*(targetSize*targetSize)+px];
              if (v > maxVal) { maxVal = v; maxClass = c; }
            }
          }
          const [r, g, b, a] = colors[maxClass] ?? [0,0,0,0];
          const i4 = px * 4;
          imgData.data[i4]=r; imgData.data[i4+1]=g; imgData.data[i4+2]=b; imgData.data[i4+3]=a;
        }
      }
      ctx.putImageData(imgData, 0, 0);
      setStatus("success");
    } catch (err) {
      console.error(err);
      setStatus("idle");
    }
  };

  const resetAll = () => { setSelectedImage(null); setStatus("idle"); };

  const segments = [
    { label: "Right Ventricle", abbr: "RV", color: "rgba(255,255,255,0.3)", sub: "Pulmonary circulation" },
    { label: "Myocardium", abbr: "MYO", color: "rgba(255,255,255,0.6)", sub: "Cardiac muscle wall" },
    { label: "Left Ventricle", abbr: "LV", color: "rgba(255,255,255,1)", sub: "Systemic circulation" },
  ];

  return (
    <div className="app">
      {/* TOP NAV */}
      <nav className="topbar">
        <div className="topbar-inner">
          <div className="logo">
            <div className="logo-mark">
              <Heart size={20} />
            </div>
            <div className="logo-text">
              <span className="name">OpenCardiac AI</span>
              <span className="tagline">Cardiac MRI Segmentation</span>
            </div>
          </div>
          <div className="topbar-right">
            <span className="version-chip">U-Net · ACDC</span>
            <div className={`status-pill ${engineReady ? 'ready' : 'loading'}`}>
              {engineReady && <span className="status-dot" />}
              {engineReady ? 'Engine Ready' : 'Loading Model...'}
            </div>
          </div>
        </div>
      </nav>

      {/* HERO */}
      <div className="hero fade-in-1">
        <div className="hero-label">
          <Zap size={13} /> AI-Powered Segmentation
        </div>
        <h1>
          Automated Cardiac{' '}
          <span className="gradient-text">Structure Analysis</span>
        </h1>
        <p>
          Upload a 2D cardiac MRI slice and let the deep learning U-Net model
          precisely segment the RV, Myocardium, and LV structures in real time — 
          entirely in your browser.
        </p>
      </div>

      {/* MAIN GRID */}
      <div className="content">
        <div className="grid-layout">
          {/* LEFT: Input + Viewer */}
          <div className="panel fade-in-2">
            <div className="panel-header">
              <div className="panel-header-left">
                <div className="panel-icon">
                  <ImageUp size={18} />
                </div>
                <div className="panel-title-wrap">
                  <h3>MRI Input</h3>
                  <p>Upload a cardiac cine MRI slice (PNG, JPG)</p>
                </div>
              </div>
              {selectedImage && (
                <button className="btn btn-clear" onClick={resetAll} style={{ padding: '0.4rem 0.75rem', fontSize: '0.75rem' }}>
                  <Trash2 size={14} /> Clear
                </button>
              )}
            </div>

            <div className="panel-body">
              <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                onChange={e => e.target.files?.[0] && loadFile(e.target.files[0])}
                style={{ display: 'none' }}
              />
              <canvas ref={hiddenCanvasRef} style={{ display: 'none' }} />

              {!selectedImage ? (
                <div
                  className={`dropzone ${isDragOver ? 'drag-active' : ''}`}
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
                  onDragLeave={() => setIsDragOver(false)}
                  onDrop={handleDrop}
                >
                  <div className="dropzone-icon-wrap">
                    <ImageUp size={32} />
                  </div>
                  <div className="dropzone-text">
                    <h4>Drop your MRI slice here</h4>
                    <p>or click to browse — PNG, JPG supported</p>
                  </div>
                </div>
              ) : (
                <>
                  <div className="viewer-wrap">
                    <img id="mri-img" src={selectedImage} alt="MRI input" />
                    <canvas ref={canvasRef} />

                    {status === 'success' && (
                      <div className="viewer-overlay-label">AI Mask Overlay</div>
                    )}

                    {status === 'processing' && (
                      <>
                        <div className="scanline-overlay">
                          <div className="scanline" />
                        </div>
                        <div className="viewer-spinner-overlay">
                          <div className="spinner-ring" />
                          <p>Analyzing structures...</p>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="controls">
                    <button
                      className="btn btn-run"
                      onClick={runPrediction}
                      disabled={status === 'processing' || !engineReady}
                    >
                      {status === 'processing' ? (
                        <><span className="spinner-ring" style={{ width: 16, height: 16, borderWidth: 2 }} />Processing</>
                      ) : (
                        <><Play size={16} fill="currentColor" />Run Segmentation</>
                      )}
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* RIGHT: Sidebar */}
          <div className="sidebar">
            {/* Status Card */}
            <div className={`status-card fade-in-2 ${status}`} style={{ animationDelay: '0.25s' }}>
              <div className="status-card-header">
                <h4>Diagnostic Status</h4>
                <Activity size={16} color={status === 'success' ? '#fff' : status === 'processing' ? '#aaa' : '#555'} />
              </div>
              <div className={`status-value ${status}`}>
                {status === 'idle' && 'Awaiting Input'}
                {status === 'processing' && 'Analyzing...'}
                {status === 'success' && '✓ Scan Complete'}
              </div>
              <div className="status-message">
                {status === 'idle' && 'Upload an MRI slice to begin analysis.'}
                {status === 'processing' && 'Running ONNX inference via WebAssembly.'}
                {status === 'success' && 'Cardiac structures successfully identified.'}
              </div>
            </div>

            {/* Segmentation Legend */}
            <div className="legend-card fade-in-3">
              <div className="legend-header">
                <Layers size={16} />
                <h4>Segmentation Classes</h4>
              </div>
              <div className="legend-items">
                {segments.map((s) => (
                  <div key={s.abbr} className={`legend-row ${status === 'success' ? 'active' : ''}`}>
                    <div className="legend-row-left">
                      <div className="legend-swatch" style={{ background: s.color }} />
                      <div>
                        <div className="legend-label">{s.label} <span style={{ color: 'var(--text-muted)' }}>({s.abbr})</span></div>
                        <div className="legend-sublabel">{s.sub}</div>
                      </div>
                    </div>
                    <div className={`legend-status ${status === 'success' ? 'detected' : 'pending'}`}>
                      {status === 'success' ? 'Detected' : '—'}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Info */}
            <div className="model-card fade-in-3" style={{ animationDelay: '0.4s' }}>
              <div className="legend-header" style={{ borderBottom: '1px solid var(--border-subtle)', marginBottom: 0, paddingLeft: 0, paddingTop: 0 }}>
                <Cpu size={15} />
                <h4>Model Info</h4>
              </div>
              <div style={{ padding: '0.5rem 0 0 0' }}>
                <div className="model-card-row">
                  <span className="key">Architecture</span>
                  <span className="val" style={{ color: '#fff' }}>U-Net (ACDC)</span>
                </div>
                <div className="model-card-row">
                  <span className="key">Format</span>
                  <span className="val">ONNX Runtime</span>
                </div>
                <div className="model-card-row">
                  <span className="key">Input Shape</span>
                  <span className="val">1 × 1 × 224 × 224</span>
                </div>
                <div className="model-card-row">
                  <span className="key">Classes</span>
                  <span className="val">4 (BG, RV, MYO, LV)</span>
                </div>
                <div className="model-card-row">
                  <span className="key">Backend</span>
                  <span className="val">WebAssembly (WASM)</span>
                </div>
                <div className="model-card-row">
                  <span className="key">Engine Status</span>
                  <span className={`val ${engineReady ? 'green' : ''}`}>{engineReady ? '● Online' : '○ Loading'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
