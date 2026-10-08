# --- PLEASE RUN THIS FIRST IN A SEPARATE CELL IN COLAB ---
# !pip install segmentation-models-pytorch timm onnx onnxscript
# --------------------------------------------------------

import torch
import segmentation_models_pytorch as smp
import onnx
import os

print("1. Initializing UnetPlusPlus architecture...")
model = smp.UnetPlusPlus(
    encoder_name="timm-tf_efficientnet_lite0",
    encoder_weights=None,
    in_channels=1,
    classes=4  # BG, RV, MYO, LV
)

print("2. Loading weights from checkpoint...")
checkpoint = torch.load('model_fixed.pt', map_location='cpu')
model_weights = checkpoint['model_state_dict']

clean_weights = {}
for k, v in model_weights.items():
    if k.startswith("model."):
        clean_weights[k[6:]] = v
    else:
        clean_weights[k] = v

model.load_state_dict(clean_weights)
model.eval()

print("3. Exporting to ONNX...")
dummy_input = torch.randn(1, 1, 224, 224)

# We use fallback=True (if available) or standard export to avoid Dynamo issues
torch.onnx.export(
    model, 
    dummy_input, 
    "cine_model_temp.onnx", 
    export_params=True, 
    opset_version=14, 
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("4. Repackaging ONNX into a single file for the browser...")
# PyTorch > 2.0 sometimes incorrectly splits weights into a separate `.onnx.data` file.
# The WebAssembly browser engine HATES split files. This merges them back together.
onnx_model = onnx.load("cine_model_temp.onnx")
onnx.save_model(onnx_model, "cine_model_new.onnx", save_as_external_data=False)

# Clean up the temp files so it's not confusing
if os.path.exists("cine_model_temp.onnx"):
    os.remove("cine_model_temp.onnx")
if os.path.exists("cine_model_temp.onnx.data"):
    os.remove("cine_model_temp.onnx.data")

print("✅ SUCCESS! Download cine_model_new.onnx from Colab and put it in your public/ folder.")
