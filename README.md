<div align="center">
  <h1>🫀 OpenCardiac AI</h1>
  <p><b>Advanced Cardiac MRI Analysis Platform</b></p>
</div>

An open-source, purely web-based platform for automated cardiac MRI segmentation and analysis. Built to bridge the gap between AI models and clinical accessibility, it runs complex deep-learning inference entirely in your browser using WebAssembly.

![UI Preview](https://via.placeholder.com/800x400.png?text=OpenCardiac+AI+-+Dashboard) *(Replace this with a real screenshot of the app!)*

## ✨ Features

- **🧠 Fully Local AI Inference:** Uses `onnxruntime-web` to segment the Left Ventricle (LV), Right Ventricle (RV), and Myocardium instantly in the browser—no backend GPUs required!
- **📂 Universal Format Support:** Seamlessly upload standard 2D DICOMs (`.dcm`) or large 3D/4D NIfTI volumes (`.nii` / `.nii.gz`).
- **⏱️ Automated Ejection Fraction (EF):** Automatically scrubs through 4D Cine scans to calculate EF curves and identify End-Diastole / End-Systole phases.
- **📊 Advanced Clinical Tools:** 
  - Interactive Windowing (WW/WL) controls.
  - Real-world volume estimation in milliliters (ml) using voxel spacing metadata.
  - Multi-slice AI averaging for noise reduction.
  - 12-sector myocardial wall thickness polar maps.
  - AI confidence heatmaps.
- **📄 One-Click Export:** Generate clinical-grade PDF reports of your segmentation and diagnostic metrics.

## 🛠️ Tech Stack

- **Frontend:** Next.js 16 (App Router), React, TypeScript
- **AI Engine:** ONNX Runtime Web (WASM)
- **Medical Parsing:** `daikon` (DICOM) & `nifti-reader-js` (NIfTI)
- **Styling:** Custom CSS with Lucide Icons

## 🚀 Quick Start

1. **Clone the repo:**
   ```bash
   git clone https://github.com/rksingh-dev/opencardiacai.git
   cd opencardiacai
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```

4. **Open in browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🩺 How It Works
The system uses a U-Net model trained on the ACDC dataset. The model has been exported to ONNX format (`public/cine_model.onnx`). When a scan is uploaded, it is normalized on-the-fly and passed as a `1x1x224x224` tensor to the WASM backend for pixel-perfect multi-class segmentation.

---
*Disclaimer: OpenCardiac AI is built for research and educational purposes only. It is not a certified medical device and should not be used as a substitute for professional clinical diagnosis.*
