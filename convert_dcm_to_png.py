import pydicom
import nibabel as nib
import numpy as np
from PIL import Image
import argparse
import sys
import os

def convert_medical_to_png(input_path, output_path, slice_idx=None):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    print(f"Loading medical image: {input_path}...")
    
    # 1. Handle DICOM format (.dcm)
    if input_path.lower().endswith('.dcm'):
        dataset = pydicom.dcmread(input_path)
        pixel_array = dataset.pixel_array
        
    # 2. Handle NIfTI format (.nii or .nii.gz)
    elif input_path.lower().endswith('.nii') or input_path.lower().endswith('.nii.gz'):
        try:
            img = nib.load(input_path)
        except Exception as e:
            print("Error: To use NIfTI files, you must install nibabel: pip install nibabel")
            sys.exit(1)
            
        img_data = img.get_fdata()
        
        # NIfTI volumes are 3D (x, y, z) or 4D (x, y, z, time)
        if len(img_data.shape) >= 3:
            # Default to extracting the exact middle slice along the Z-axis
            if slice_idx is None:
                slice_idx = img_data.shape[2] // 2
            
            # If 4D (like ACDC cine MRI), just grab the first time frame
            if len(img_data.shape) == 4:
                pixel_array = img_data[:, :, slice_idx, 0]
            else:
                pixel_array = img_data[:, :, slice_idx]
                
            print(f"Extracted slice {slice_idx} from 3D volume.")
        else:
            pixel_array = img_data
            
        # Rotate NIfTI 90 degrees to match typical viewer orientation
        pixel_array = np.rot90(pixel_array)
        
    else:
        print("Error: Unsupported format. Please provide a .dcm or .nii.gz file.")
        sys.exit(1)
    
    # 3. Normalize the pixel array to standard 8-bit (0-255)
    img_min = np.min(pixel_array)
    img_max = np.max(pixel_array)
    
    if img_max == img_min:
        print("Warning: Image is completely blank.")
        normalized_img = pixel_array.astype(np.uint8)
    else:
        normalized_img = ((pixel_array - img_min) / (img_max - img_min)) * 255.0
        normalized_img = normalized_img.astype(np.uint8)
    
    # 4. Save to PNG
    img = Image.fromarray(normalized_img)
    img.save(output_path)
    
    print(f"Success! Saved PNG to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .dcm or .nii.gz file to a 2D .png image.")
    parser.add_argument("input_path", help="Path to the input .dcm or .nii.gz file")
    parser.add_argument("output_png", help="Path to save the output .png file")
    parser.add_argument("--slice", type=int, default=None, help="Z-axis slice index to extract from 3D NIfTI files (defaults to middle slice)")
    
    args = parser.parse_args()
    convert_medical_to_png(args.input_path, args.output_png, args.slice)
