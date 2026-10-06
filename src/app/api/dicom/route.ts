import { NextRequest, NextResponse } from 'next/server';
import * as daikon from 'daikon';
import * as nifti from 'nifti-reader-js';

function normalizeToUint8(pixels: ArrayLike<number>): Uint8Array {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < pixels.length; i++) {
    if (pixels[i] < min) min = pixels[i];
    if (pixels[i] > max) max = pixels[i];
  }
  const range = max - min || 1;
  const out = new Uint8Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    out[i] = ((pixels[i] - min) / range) * 255;
  }
  return out;
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    if (!file) return NextResponse.json({ error: "No file provided" }, { status: 400 });

    const buffer = await file.arrayBuffer() as ArrayBuffer;
    const filename = file.name.toLowerCase();

    // ═══════════════════════════════════════════════════════════
    // NIfTI path: .nii / .nii.gz
    // Returns ALL slices for Z-scrubber + pixdim for volume calc
    // ═══════════════════════════════════════════════════════════
    if (filename.endsWith('.nii') || filename.endsWith('.nii.gz')) {
      let niftiBuffer: ArrayBuffer = buffer;
      if (nifti.isCompressed(buffer)) {
        niftiBuffer = nifti.decompress(buffer) as ArrayBuffer;
      }
      if (!nifti.isNIFTI(niftiBuffer)) {
        throw new Error("Invalid NIfTI file format.");
      }

      const niftiHeader = nifti.readHeader(niftiBuffer);
      const niftiImage = nifti.readImage(niftiHeader, niftiBuffer);

      const cols = niftiHeader.dims[1];
      const rows = niftiHeader.dims[2];
      const slices = niftiHeader.dims[3] || 1;
      const timeFrames = niftiHeader.dims[4] || 1;

      // Feature 4: Voxel dimensions in mm (for volume calculation)
      const pixdim = [
        Math.abs(niftiHeader.pixDims[1]) || 1,
        Math.abs(niftiHeader.pixDims[2]) || 1,
        Math.abs(niftiHeader.pixDims[3]) || 1,
      ];

      let typedData: ArrayLike<number>;
      if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT8)    typedData = new Uint8Array(niftiImage);
      else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT16)   typedData = new Int16Array(niftiImage);
      else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT32)   typedData = new Int32Array(niftiImage);
      else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) typedData = new Float32Array(niftiImage);
      else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT64) typedData = new Float64Array(niftiImage);
      else throw new Error("Unsupported NIfTI datatype: " + niftiHeader.datatypeCode);

      const sliceSize = cols * rows;
      const is4D = timeFrames > 1;
      const allSlicesB64: string[] = [];

      if (is4D) {
        // 4D Cine MRI: all time frames at the middle Z-slice (for EF calc)
        const midSlice = Math.floor(slices / 2);
        for (let t = 0; t < timeFrames; t++) {
          const offset = (t * slices * sliceSize) + (midSlice * sliceSize);
          const raw = Array.from({ length: sliceSize }, (_, i) => (typedData as any)[offset + i]);
          allSlicesB64.push(Buffer.from(normalizeToUint8(raw)).toString('base64'));
        }
        return NextResponse.json({ cols, rows, sliceCount: timeFrames, mode: '4d_cine', slices: allSlicesB64, pixdim });
      } else {
        // 3D volume: all Z-axis slices for the Z-scrubber
        for (let s = 0; s < slices; s++) {
          const raw = Array.from({ length: sliceSize }, (_, i) => (typedData as any)[s * sliceSize + i]);
          allSlicesB64.push(Buffer.from(normalizeToUint8(raw)).toString('base64'));
        }
        return NextResponse.json({ cols, rows, sliceCount: slices, mode: '3d_volume', slices: allSlicesB64, pixdim });
      }
    }

    // ════════════════════════════
    // DICOM path: single 2D slice
    // ════════════════════════════
    const dataView = new DataView(buffer as ArrayBuffer);
    const image = daikon.Series.parseImage(dataView);
    if (!image) throw new Error("Could not parse DICOM image.");

    const cols = image.getCols();
    const rows = image.getRows();
    const pixels = image.getInterpretedData();
    const base64Str = Buffer.from(normalizeToUint8(pixels)).toString('base64');

    // Try to get pixel spacing from DICOM tags
    const pixelSpacing = image.getPixelSpacing?.() ?? [1, 1];
    const sliceThickness = image.getSliceThickness?.() ?? 1;
    const pixdim = [pixelSpacing[0] || 1, pixelSpacing[1] || 1, sliceThickness || 1];

    return NextResponse.json({ cols, rows, sliceCount: 1, mode: 'dicom', slices: [base64Str], pixdim });

  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
