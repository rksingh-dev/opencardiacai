# 🫀 OpenCardiac AI Studio — CMRI Upload & Segmentation Pipeline

An interactive, clinical-grade medical AI system built with **Streamlit** and **FastAPI** to upload Cardiac MRI (CMRI) scans, run deep learning segmentation with **`data.pkl`**, and export/send the segmentation results.

---

## 🚀 Quick Start

### 1. Web Application (Streamlit)
Double-click:
```cmd
run_app.bat
```
Or run:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 2. REST API Server (FastAPI)
Double-click:
```cmd
run_api.bat
```
Or run:
```bash
python -m uvicorn api_server:app --port 8000
```
Interactive Swagger Documentation available at **`http://localhost:8000/docs`**.

---

## 🏥 End-to-End Workflow: Upload CMRI ➔ Send to Model ➔ Receive Segmentation

### Mode A: Interactive Web UI (Streamlit at `localhost:8501`)
1. **Step 1: Upload CMRI**
   - Drag & drop your Cardiac MRI scan:
     - **DICOM** (`.dcm`) — Auto-extracts Pixel Spacing, Slice Thickness, and Patient ID.
     - **NIfTI 3D/4D** (`.nii`, `.nii.gz`) — Multi-slice Z-scrubber slider.
     - **Images** (`.png`, `.jpg`, `.jpeg`).
   - Or pick preloaded clinical demo slices (End-Diastole, End-Systole, Myocardial Hypertrophy).
2. **Step 2: Send to Model**
   - Click the prominent **`🚀 Send to Model & Run Segmentation`** button.
   - Dispatches the scan to the `UnetPlusPlus` model (`data.pkl` checkpoint).
3. **Step 3: Review Segmentation & Clinical Biometrics**
   - Side-by-side display:
     - Original Scan
     - Color-Coded Overlay (Customizable opacity)
     - Segmentation Mask
     - Isolated Left Ventricle (LV)
   - Real-time clinical parameters:
     - Left Ventricle (LV) Area ($cm^2$)
     - Myocardium (MYO) Area ($cm^2$)
     - Right Ventricle (RV) Area ($cm^2$)
     - MYO / LV Ratio with diagnostic alert
     - 12-Sector Myocardial Wall Thickness Polar Map
     - Automated Clinical Impression
4. **Step 4: Send & Export Segmentation**
   - 📥 **Download Mask (PNG)**
   - 📥 **Download Overlay (PNG)**
   - 📥 **Download Full Bundle (.ZIP)** (Includes all binary masks for LV, MYO, RV, plus JSON clinical report)
   - 📥 **Export Clinical Report (JSON)**
   - 🌐 **Send to External Webhook / PACS**: Enter a remote server URL and click **`📡 Send Segmentation Now`** to POST the findings and base64 masks directly.

---

### Mode B: Programmatic REST API (`localhost:8000`)

#### 1. Upload CMRI and Get Full Segmentation JSON
```bash
curl -X POST "http://localhost:8000/api/segment" \
     -F "file=@my_scan.dcm"
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/api/segment"
files = {"file": open("patient_scan.dcm", "rb")}
response = requests.post(url, files=files)
data = response.json()

print("Status:", data["status"])
print("Diagnosis:", data["metrics"]["diagnosis"])
print("LV Area:", data["metrics"]["lv_area_cm2"], "cm²")
print("MYO/LV Ratio:", data["metrics"]["myo_lv_ratio"])

# Base64 encoded segmentation PNGs:
mask_b64 = data["segmentation"]["mask_png_base64"]
overlay_b64 = data["segmentation"]["overlay_png_base64"]
```

#### 2. Get Raw Segmentation Mask PNG directly:
```bash
curl -X POST "http://localhost:8000/api/segment/mask" \
     -F "file=@my_scan.nii.gz" \
     --output predicted_mask.png
```

#### 3. Download Full Segmentation ZIP Bundle directly:
```bash
curl -X POST "http://localhost:8000/api/segment/zip" \
     -F "file=@my_scan.dcm" \
     --output cardiac_bundle.zip
```

---

## 📦 Deep Learning Checkpoint Details (`data.pkl`)

- **Model Architecture:** `UnetPlusPlus` (Nested Dense Skip Pathways)
- **Backbone Encoder:** `timm-tf_efficientnet_lite0`
- **Validation Dice Score:** **`0.9267` (92.67%)**
- **Validation Loss:** `0.0995`
- **Epoch / Fold:** Epoch 29, Fold 0
- **Total Parameters:** `5,977,707` (~6M parameters)
- **Target Classes:**
  - `0`: Background
  - `1`: Left Ventricle (LV) Endocardium / Cavity (Red)
  - `2`: Myocardium (MYO) Muscle (Emerald)
  - `3`: Right Ventricle (RV) Cavity (Blue)

---

*Disclaimer: For research and educational purposes only. Not certified as a medical device.*
