<div align="center">
  <h1>🫀 OpenCardiac AI</h1>
  <p><b>Web-Based Cardiac MRI Segmentation & Analysis</b></p>
</div>

OpenCardiac AI is a browser-based clinical tool that uses an ONNX-exported U-Net model to instantly segment cardiac MRI scans without requiring a backend GPU.

## ✨ Features

- **Local Inference:** Fully private, in-browser AI processing using WebAssembly.
- **Format Support:** Supports 2D DICOM (`.dcm`) and 3D/4D NIfTI (`.nii.gz`) volumes.
- **Clinical Tools:** 
  - Automated Ejection Fraction (EF) calculation on 4D Cine scans.
  - Myocardial wall thickness mapping (12-sector polar chart).
  - Voxel-based real-world volume estimation (ml).
  - AI confidence heatmaps and multi-slice averaging.
- **Export:** One-click PDF report generation.


*Disclaimer: For research and educational purposes only. Not a medical device.*
