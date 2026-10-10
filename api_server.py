"""
api_server.py - High-Performance REST API for OpenCardiac AI
Allows external applications, EHR, PACS, or frontend clients to:
Upload CMRI (DICOM, NIfTI, or Image) -> Send to Model -> Receive Segmentation & Clinical Metrics.
"""

import os
import io
import time
import base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import model_utils

app = FastAPI(
    title="OpenCardiac AI - CMRI Segmentation API",
    description="Upload CMRI (DICOM / NIfTI / PNG) -> Run Deep Learning Segmentation -> Return Segmentations & Biometrics",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache inference engine
engine = model_utils.get_inference_engine()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "OpenCardiac AI CMRI Segmentation API",
        "model": "UnetPlusPlus (timm-tf_efficientnet_lite0)",
        "engine": engine["type"].upper() if engine else "OFFLINE",
        "docs_url": "/docs",
        "endpoints": {
            "segment_json": "POST /api/segment",
            "segment_mask_image": "POST /api/segment/mask",
            "segment_overlay_image": "POST /api/segment/overlay",
            "segment_zip_bundle": "POST /api/segment/zip"
        }
    }

@app.post("/api/segment")
async def segment_cmri(
    file: UploadFile = File(..., description="CMRI file (.dcm, .nii, .nii.gz, .png, .jpg)"),
    spacing_x: float = Form(1.4),
    spacing_y: float = Form(1.4),
    alpha: float = Form(0.52),
    slice_idx: Optional[int] = Form(None),
    high_res: bool = Form(True),
    palette: str = Form("classic")
):
    """
    Upload a CMRI file, run segmentation inference, and return structured JSON
    including clinical biometrics and base64-encoded segmentation masks.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded.")

    contents = await file.read()
    try:
        parsed = model_utils.parse_cmri(contents, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CMRI file: {str(e)}")

    slices = parsed["slices"]
    if len(slices) == 0:
        raise HTTPException(status_code=400, detail="No readable slices found in CMRI file.")

    # Select target slice
    target_idx = slice_idx if (slice_idx is not None and 0 <= slice_idx < len(slices)) else len(slices) // 2
    raw_slice = slices[target_idx]

    # Preprocess & Inference
    t0 = time.time()
    input_tensor, norm_img_224 = model_utils.preprocess_medical_image(raw_slice)
    mask, probs = model_utils.run_inference(engine, input_tensor)
    inference_ms = (time.time() - t0) * 1000

    # Spacing resolution
    sx = parsed["pixdim"][0] if parsed["pixdim"][0] > 0 else spacing_x
    sy = parsed["pixdim"][1] if parsed["pixdim"][1] > 0 else spacing_y
    metrics = model_utils.compute_clinical_metrics(mask, (sx, sy))

    # Package output with High-Res if requested
    if high_res:
        hd = model_utils.create_high_res_color_overlay(
            base_img_orig=raw_slice,
            probs_224=probs,
            alpha=alpha,
            palette=palette
        )
        pkg = model_utils.generate_segmentation_package(
            norm_img_224, mask, metrics, alpha=alpha,
            high_res_overlay_uint8=hd["composite_uint8"],
            high_res_mask=hd["mask_high"],
            palette=palette
        )
    else:
        pkg = model_utils.generate_segmentation_package(norm_img_224, mask, metrics, alpha=alpha, palette=palette)

    return {
        "status": "success",
        "filename": file.filename,
        "format": parsed["mode"],
        "total_slices": len(slices),
        "selected_slice_index": target_idx,
        "inference_time_ms": round(inference_ms, 2),
        "pixel_spacing_mm": [sx, sy],
        "metrics": metrics,
        "segmentation": {
            "mask_png_base64": base64.b64encode(pkg["mask_bytes"]).decode("utf-8"),
            "overlay_png_base64": base64.b64encode(pkg["overlay_bytes"]).decode("utf-8"),
            "raw_labels_png_base64": base64.b64encode(pkg["raw_mask_bytes"]).decode("utf-8"),
            "lv_mask_base64": base64.b64encode(pkg["lv_mask_bytes"]).decode("utf-8"),
            "myo_mask_base64": base64.b64encode(pkg["myo_mask_bytes"]).decode("utf-8"),
            "rv_mask_base64": base64.b64encode(pkg["rv_mask_bytes"]).decode("utf-8")
        },
        "classes": {
            "0": "Background",
            "1": "Right Ventricle (RV)",
            "2": "Myocardium (MYO)",
            "3": "Left Ventricle (LV)"
        }
    }

@app.post("/api/segment/mask")
async def get_raw_mask(file: UploadFile = File(...), high_res: bool = Form(True)):
    """Returns the segmentation mask directly as an image/png."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded.")
    contents = await file.read()
    parsed = model_utils.parse_cmri(contents, filename=file.filename)
    raw_slice = parsed["slices"][len(parsed["slices"]) // 2]
    input_tensor, norm_img_224 = model_utils.preprocess_medical_image(raw_slice)
    mask, probs = model_utils.run_inference(engine, input_tensor)
    metrics = model_utils.compute_clinical_metrics(mask)
    if high_res:
        hd = model_utils.create_high_res_color_overlay(base_img_orig=raw_slice, probs_224=probs)
        pkg = model_utils.generate_segmentation_package(norm_img_224, mask, metrics, high_res_overlay_uint8=hd["composite_uint8"], high_res_mask=hd["mask_high"])
    else:
        pkg = model_utils.generate_segmentation_package(norm_img_224, mask, metrics)
    return Response(content=pkg["mask_bytes"], media_type="image/png")

@app.post("/api/segment/overlay")
async def get_raw_overlay(file: UploadFile = File(...), alpha: float = Form(0.52), high_res: bool = Form(True)):
    """Returns the color-coded overlay directly as an image/png."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded.")
    contents = await file.read()
    parsed = model_utils.parse_cmri(contents, filename=file.filename)
    raw_slice = parsed["slices"][len(parsed["slices"]) // 2]
    input_tensor, norm_img_224 = model_utils.preprocess_medical_image(raw_slice)
    mask, probs = model_utils.run_inference(engine, input_tensor)
    metrics = model_utils.compute_clinical_metrics(mask)
    if high_res:
        hd = model_utils.create_high_res_color_overlay(base_img_orig=raw_slice, probs_224=probs, alpha=alpha)
        pkg = model_utils.generate_segmentation_package(norm_img_224, mask, metrics, alpha=alpha, high_res_overlay_uint8=hd["composite_uint8"], high_res_mask=hd["mask_high"])
    else:
        pkg = model_utils.generate_segmentation_package(norm_img_224, mask, metrics, alpha=alpha)
    return Response(content=pkg["overlay_bytes"], media_type="image/png")

@app.post("/api/segment/zip")
async def get_zip_bundle(file: UploadFile = File(...), high_res: bool = Form(True)):
    """Returns a complete in-memory ZIP package containing all masks and JSON report."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded.")
    contents = await file.read()
    parsed = model_utils.parse_cmri(contents, filename=file.filename)
    raw_slice = parsed["slices"][len(parsed["slices"]) // 2]
    input_tensor, norm_img_224 = model_utils.preprocess_medical_image(raw_slice)
    mask, probs = model_utils.run_inference(engine, input_tensor)
    metrics = model_utils.compute_clinical_metrics(mask)
    if high_res:
        hd = model_utils.create_high_res_color_overlay(base_img_orig=raw_slice, probs_224=probs)
        pkg = model_utils.generate_segmentation_package(norm_img_224, mask, metrics, high_res_overlay_uint8=hd["composite_uint8"], high_res_mask=hd["mask_high"])
    else:
        pkg = model_utils.generate_segmentation_package(norm_img_224, mask, metrics)
    return Response(
        content=pkg["zip_bytes"],
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=cardiac_segmentation_{file.filename}.zip"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
