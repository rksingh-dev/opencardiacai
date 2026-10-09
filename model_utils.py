"""
model_utils.py - Core ML and Data Handling Engine for OpenCardiacAI
Supports loading data.pkl (PyTorch state dict & metadata), model_fixed.pt,
and public/cine_model.onnx for cardiac MRI inference and clinical metrics calculation.
"""

import os
import io
import time
import zipfile
import base64
import json
import pickle
import numpy as np
from PIL import Image
import requests

# Custom Unpickler to safely load PyTorch persistent storages saved via pickle
class TorchCheckpointUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        try:
            storage_type, storage_class, key, location, size = pid
            return storage_class(size)
        except Exception:
            if isinstance(pid, (tuple, list)) and len(pid) >= 5:
                return pid[1](pid[4])
            raise

def load_data_pkl(filepath="data.pkl"):
    """
    Safely load data.pkl, supporting both standard torch checkpoint and
    pickle streams with PyTorch persistent storage references.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        import torch
        data = torch.load(filepath, map_location="cpu")
        return data
    except Exception:
        pass

    with open(filepath, "rb") as f:
        unpickler = TorchCheckpointUnpickler(f)
        data = unpickler.load()
        return data

def parse_checkpoint_metadata(checkpoint):
    """
    Extract training metadata and layer architecture details from checkpoint.
    """
    if not isinstance(checkpoint, dict):
        return {
            "type": str(type(checkpoint)),
            "is_dict": False,
            "raw_str": str(checkpoint)[:1000]
        }

    meta = {
        "fold": checkpoint.get("fold", "N/A"),
        "epoch": checkpoint.get("epoch", "N/A"),
        "best_val_dice": checkpoint.get("best_val_dice", None),
        "best_val_loss": checkpoint.get("best_val_loss", None),
        "model": checkpoint.get("model", "UnetPlusPlus"),
        "encoder": checkpoint.get("encoder", "timm-tf_efficientnet_lite0"),
        "learning_rate": checkpoint.get("learning_rate", 0.001),
        "batch_size": checkpoint.get("batch_size", 32),
        "keys": list(checkpoint.keys()),
    }

    state_dict = None
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    layer_info = []
    total_params = 0
    encoder_params = 0
    decoder_params = 0
    head_params = 0

    if isinstance(state_dict, dict):
        for name, tensor in state_dict.items():
            shape = tuple(tensor.shape) if hasattr(tensor, "shape") else ()
            num_elem = int(np.prod(shape)) if shape else 1
            dtype_str = str(getattr(tensor, "dtype", "unknown"))

            category = "Other"
            if "encoder" in name:
                category = "Encoder"
                encoder_params += num_elem
            elif "decoder" in name:
                category = "Decoder"
                decoder_params += num_elem
            elif "segmentation_head" in name or "head" in name:
                category = "Head"
                head_params += num_elem

            total_params += num_elem
            layer_info.append({
                "layer_name": name,
                "shape": str(list(shape)),
                "num_elements": num_elem,
                "dtype": dtype_str,
                "category": category
            })

    meta["total_parameters"] = total_params
    meta["encoder_parameters"] = encoder_params
    meta["decoder_parameters"] = decoder_params
    meta["head_parameters"] = head_params
    meta["total_layers"] = len(layer_info)
    meta["layers"] = layer_info

    return meta

# Inference engine loader (supports ONNX Runtime and PyTorch)
_INFERENCE_SESSION = None

def get_inference_engine():
    """
    Initialize an inference session. Prefers ONNX if cine_model.onnx exists,
    otherwise falls back to PyTorch SMP model if model_fixed.pt is present.
    """
    global _INFERENCE_SESSION
    if _INFERENCE_SESSION is not None:
        return _INFERENCE_SESSION

    onnx_path = os.path.join(os.path.dirname(__file__), "public", "cine_model.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = "cine_model.onnx"

    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            _INFERENCE_SESSION = {"type": "onnx", "session": sess, "path": onnx_path}
            return _INFERENCE_SESSION
        except Exception as e:
            print("Failed to initialize ONNXRuntime session:", e)

    pt_path = "model_fixed.pt"
    if os.path.exists(pt_path):
        try:
            import torch
            import segmentation_models_pytorch as smp
            model = smp.UnetPlusPlus(
                encoder_name="timm-tf_efficientnet_lite0",
                encoder_weights=None,
                in_channels=1,
                classes=4
            )
            ckpt = torch.load(pt_path, map_location="cpu")
            sd = ckpt["model_state_dict"]
            clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
            model.load_state_dict(clean_sd, strict=False)
            model.eval()
            _INFERENCE_SESSION = {"type": "torch", "model": model, "path": pt_path}
            return _INFERENCE_SESSION
        except Exception as e:
            print("Failed to initialize PyTorch model:", e)

    return None

def parse_cmri(file_bytes_or_path, filename="scan.dcm"):
    """
    Parses an uploaded CMRI file in DICOM (.dcm), NIfTI (.nii, .nii.gz), or image formats.
    Returns:
    {
        "mode": "dicom" | "3d_volume" | "4d_cine" | "image",
        "slices": [list of 2D float32 numpy arrays],
        "pixdim": [spacing_x, spacing_y, slice_thickness],
        "metadata": dict
    }
    """
    fname_lower = filename.lower()
    metadata = {"filename": filename}

    # Helper: read buffer
    if isinstance(file_bytes_or_path, (str, os.PathLike)):
        with open(file_bytes_or_path, "rb") as f:
            buf = f.read()
    elif hasattr(file_bytes_or_path, "read"):
        buf = file_bytes_or_path.read()
    else:
        buf = file_bytes_or_path

    # 1. DICOM (.dcm)
    if fname_lower.endswith(".dcm"):
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(buf))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Spacing
        spacing = [1.4, 1.4, 8.0]
        if hasattr(ds, "PixelSpacing"):
            spacing[0] = float(ds.PixelSpacing[0])
            spacing[1] = float(ds.PixelSpacing[1])
        if hasattr(ds, "SliceThickness"):
            spacing[2] = float(ds.SliceThickness)

        metadata["patient_id"] = str(getattr(ds, "PatientID", "Anonymous"))
        metadata["study_description"] = str(getattr(ds, "StudyDescription", "Cardiac MRI"))
        metadata["series_description"] = str(getattr(ds, "SeriesDescription", "Short Axis Cine"))
        metadata["rows"] = int(getattr(ds, "Rows", pixel_array.shape[0]))
        metadata["cols"] = int(getattr(ds, "Columns", pixel_array.shape[1]))

        # Normalize 2D
        if pixel_array.ndim > 2:
            pixel_array = pixel_array[0]

        return {
            "mode": "dicom",
            "slices": [pixel_array],
            "pixdim": spacing,
            "metadata": metadata
        }

    # 2. NIfTI (.nii, .nii.gz)
    elif fname_lower.endswith(".nii") or fname_lower.endswith(".nii.gz") or fname_lower.endswith(".gz"):
        import nibabel as nib
        temp_path = os.path.join("samples", f"temp_{int(time.time()*1000)}_{filename}")
        os.makedirs("samples", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(buf)

        try:
            img = nib.load(temp_path)
            data = img.get_fdata().astype(np.float32)
            header = img.header
            zooms = list(header.get_zooms())
            spacing = [float(zooms[0]) if len(zooms)>0 else 1.4,
                       float(zooms[1]) if len(zooms)>1 else 1.4,
                       float(zooms[2]) if len(zooms)>2 else 8.0]

            metadata["nii_shape"] = list(data.shape)
            slices_list = []

            if data.ndim == 4:
                # 4D cine MRI (x, y, z, time) -> extract mid-slice across all time frames
                mid_z = data.shape[2] // 2
                for t in range(data.shape[3]):
                    sl = np.rot90(data[:, :, mid_z, t])
                    slices_list.append(sl)
                mode = "4d_cine"
            elif data.ndim >= 3:
                # 3D volume (x, y, z) -> extract all slices along z
                for z in range(data.shape[2]):
                    sl = np.rot90(data[:, :, z])
                    slices_list.append(sl)
                mode = "3d_volume"
            else:
                slices_list.append(np.rot90(data))
                mode = "image"

            return {
                "mode": mode,
                "slices": slices_list,
                "pixdim": spacing,
                "metadata": metadata
            }
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # 3. Standard 2D Image (.png, .jpg, etc.)
    else:
        pil_img = Image.open(io.BytesIO(buf)).convert("L")
        arr = np.array(pil_img, dtype=np.float32)
        return {
            "mode": "image",
            "slices": [arr],
            "pixdim": [1.4, 1.4, 8.0],
            "metadata": metadata
        }

def preprocess_medical_image(img_array, target_size=(224, 224)):
    """
    Normalize 2D medical slice to [0, 1] and resize to target_size (224, 224).
    Returns (1, 1, 224, 224) float32 tensor and normalized display image.
    """
    img_array = np.array(img_array, dtype=np.float32)
    if img_array.ndim > 2:
        img_array = img_array.squeeze()
        if img_array.ndim > 2:
            img_array = img_array.mean(axis=-1)

    min_val = np.min(img_array)
    max_val = np.max(img_array)
    if max_val > min_val:
        norm_img = (img_array - min_val) / (max_val - min_val)
    else:
        norm_img = np.zeros_like(img_array)

    pil_img = Image.fromarray((norm_img * 255.0).astype(np.uint8))
    pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    resized_arr = np.array(pil_resized, dtype=np.float32) / 255.0

    input_tensor = resized_arr[np.newaxis, np.newaxis, :, :].astype(np.float32)
    return input_tensor, resized_arr

def run_inference(engine, input_tensor):
    """
    Run forward pass and return predicted mask and class probabilities.
    0: Background, 1: LV, 2: MYO, 3: RV
    """
    if engine["type"] == "onnx":
        sess = engine["session"]
        input_name = sess.get_inputs()[0].name
        logits = sess.run(None, {input_name: input_tensor})[0]
    else:
        import torch
        model = engine["model"]
        with torch.no_grad():
            t = torch.from_numpy(input_tensor)
            logits = model(t).numpy()

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    mask = np.argmax(probs, axis=1)[0]
    return mask, probs[0]

def compute_clinical_metrics(mask, pixel_spacing=(1.4, 1.4)):
    """
    Compute cardiac morphological parameters and clinical impressions.
    """
    area_per_pixel_mm2 = pixel_spacing[0] * pixel_spacing[1]

    lv_pixels = int(np.sum(mask == 1))
    myo_pixels = int(np.sum(mask == 2))
    rv_pixels = int(np.sum(mask == 3))

    lv_area_cm2 = (lv_pixels * area_per_pixel_mm2) / 100.0
    myo_area_cm2 = (myo_pixels * area_per_pixel_mm2) / 100.0
    rv_area_cm2 = (rv_pixels * area_per_pixel_mm2) / 100.0

    myo_lv_ratio = (myo_pixels / lv_pixels) if lv_pixels > 0 else 0.0

    if lv_pixels == 0:
        diagnosis = "No Cardiac Structures Detected in Slice"
        status_color = "gray"
    elif myo_lv_ratio > 1.8:
        diagnosis = "Suspected Myocardial Hypertrophy (Thickened Wall)"
        status_color = "red"
    elif myo_lv_ratio < 0.6 and myo_lv_ratio > 0:
        diagnosis = "Suspected Dilated Ventricle / Wall Thinning"
        status_color = "orange"
    else:
        diagnosis = "Normal Cardiac Morphology"
        status_color = "green"

    wall_thicknesses = compute_12_sector_wall_thickness(mask, pixel_spacing)

    return {
        "lv_pixels": lv_pixels,
        "myo_pixels": myo_pixels,
        "rv_pixels": rv_pixels,
        "lv_area_cm2": round(lv_area_cm2, 2),
        "myo_area_cm2": round(myo_area_cm2, 2),
        "rv_area_cm2": round(rv_area_cm2, 2),
        "myo_lv_ratio": round(myo_lv_ratio, 2),
        "diagnosis": diagnosis,
        "status_color": status_color,
        "wall_thicknesses": wall_thicknesses
    }

def compute_12_sector_wall_thickness(mask, pixel_spacing=(1.4, 1.4)):
    """
    Calculates average myocardial wall thickness across 12 clock sectors (30 deg each).
    """
    lv_pts = np.argwhere(mask == 1)
    if len(lv_pts) < 10:
        return [0.0] * 12

    center_y, center_x = np.mean(lv_pts, axis=0)

    myo_pts = np.argwhere(mask == 2)
    if len(myo_pts) < 10:
        return [0.0] * 12

    angles = np.arctan2(myo_pts[:, 0] - center_y, myo_pts[:, 1] - center_x)
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    sector_bins = np.linspace(0, 2 * np.pi, 13)
    avg_spacing = (pixel_spacing[0] + pixel_spacing[1]) / 2.0

    sector_thicknesses = []
    for i in range(12):
        in_sector = (angles >= sector_bins[i]) & (angles < sector_bins[i + 1])
        pts_in_sec = myo_pts[in_sector]
        if len(pts_in_sec) > 0:
            radii = np.sqrt((pts_in_sec[:, 0] - center_y)**2 + (pts_in_sec[:, 1] - center_x)**2)
            thickness_mm = (np.max(radii) - np.min(radii)) * avg_spacing
            sector_thicknesses.append(round(float(max(thickness_mm, 0.0)), 2))
        else:
            sector_thicknesses.append(0.0)

    return sector_thicknesses

def create_color_overlay(base_img_224, mask, alpha=0.55):
    """
    Generate an RGB image with color-coded segmentation mask overlay.
    - Left Ventricle (LV, 1): Red [239, 68, 68]
    - Myocardium (MYO, 2): Emerald [16, 185, 129]
    - Right Ventricle (RV, 3): Electric Blue [59, 130, 246]
    """
    base_rgb = np.stack([base_img_224, base_img_224, base_img_224], axis=-1)
    overlay_rgb = base_rgb.copy()

    colors = {
        1: np.array([239/255.0, 68/255.0, 68/255.0]),
        2: np.array([16/255.0, 185/255.0, 129/255.0]),
        3: np.array([59/255.0, 130/255.0, 246/255.0]),
    }

    for class_id, color in colors.items():
        idx = (mask == class_id)
        if np.any(idx):
            overlay_rgb[idx] = (1 - alpha) * base_rgb[idx] + alpha * color

    return np.clip(overlay_rgb, 0.0, 1.0)

def array_to_png_bytes(arr_uint8):
    """Encodes a uint8 numpy array to PNG bytes in memory."""
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_segmentation_package(raw_norm_224, mask, metrics, alpha=0.55):
    """
    Builds a complete exportable package:
    - mask_png: indexed mask PNG (0, 1, 2, 3)
    - overlay_png: RGB color overlay PNG
    - lv_mask_png: binary LV cavity mask PNG (0 or 255)
    - myo_mask_png: binary Myocardium mask PNG (0 or 255)
    - rv_mask_png: binary RV cavity mask PNG (0 or 255)
    - zip_bytes: packaged ZIP containing all files + JSON report
    """
    # 1. Overlay
    overlay_rgb = create_color_overlay(raw_norm_224, mask, alpha=alpha)
    overlay_uint8 = (overlay_rgb * 255).astype(np.uint8)
    overlay_bytes = array_to_png_bytes(overlay_uint8)

    # 2. Mask (Colorized for viewing + raw label)
    mask_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    mask_rgb[mask == 1] = [239, 68, 68]
    mask_rgb[mask == 2] = [16, 185, 129]
    mask_rgb[mask == 3] = [59, 130, 246]
    mask_bytes = array_to_png_bytes(mask_rgb)
    raw_mask_label_bytes = array_to_png_bytes(mask.astype(np.uint8))

    # 3. Binary masks
    lv_mask_bytes = array_to_png_bytes((mask == 1).astype(np.uint8) * 255)
    myo_mask_bytes = array_to_png_bytes((mask == 2).astype(np.uint8) * 255)
    rv_mask_bytes = array_to_png_bytes((mask == 3).astype(np.uint8) * 255)

    # 4. JSON report
    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "clinical_metrics": metrics,
        "classes": {
            0: "Background",
            1: "Left Ventricle (LV)",
            2: "Myocardium (MYO)",
            3: "Right Ventricle (RV)"
        }
    }
    report_json_bytes = json.dumps(report_dict, indent=2).encode("utf-8")

    # 5. Pack into in-memory ZIP bundle
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("segmentation_overlay.png", overlay_bytes)
        zf.writestr("segmentation_mask_color.png", mask_bytes)
        zf.writestr("segmentation_mask_labels.png", raw_mask_label_bytes)
        zf.writestr("mask_left_ventricle.png", lv_mask_bytes)
        zf.writestr("mask_myocardium.png", myo_mask_bytes)
        zf.writestr("mask_right_ventricle.png", rv_mask_bytes)
        zf.writestr("cardiac_metrics_report.json", report_json_bytes)

    zip_bytes = zip_buf.getvalue()

    return {
        "overlay_bytes": overlay_bytes,
        "mask_bytes": mask_bytes,
        "raw_mask_bytes": raw_mask_label_bytes,
        "lv_mask_bytes": lv_mask_bytes,
        "myo_mask_bytes": myo_mask_bytes,
        "rv_mask_bytes": rv_mask_bytes,
        "report_json_bytes": report_json_bytes,
        "zip_bytes": zip_bytes
    }

def send_segmentation_to_endpoint(target_url, metrics, mask_base64=None, overlay_base64=None):
    """
    Sends the generated segmentation result and clinical metrics to an external REST / Webhook endpoint.
    """
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": metrics,
        "segmentation": {
            "mask_png_base64": mask_base64,
            "overlay_png_base64": overlay_base64
        }
    }
    resp = requests.post(target_url, json=payload, timeout=10)
    return resp.status_code, resp.text
