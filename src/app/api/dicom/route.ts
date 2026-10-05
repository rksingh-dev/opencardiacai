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

    const buffer = await file.arrayBuffer();
    const filename = file.name.toLowerCase();

    // ═══════════════════════════════════════════
    // NIfTI path: supports .nii and .nii.gz
    // Sends ALL slices back for the Z-axis scrubber
    // Also handles 4D (cine) for Ejection Fraction
    // ═══════════════════════════════════════════
    if (filename.endsWith('.nii') || filename.endsWith('.nii.gz')) {
      let niftiBuffer = buffer;
      if (nifti.isCompressed(buffer)) {
        niftiBuffer = nifti.decompress(buffer);
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

      let typedData: ArrayLike<number>;
      if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT8) typedData = new Uint8Array(niftiImage);
      else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT16) typedData = new Int16Array(niftiImage);
      else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT32) typedData = new Int32Array(niftiImage);
      else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) typedData = new Float32Array(niftiImage);
      else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT64) typedData = new Float64Array(niftiImage);
      else throw new Error("Unsupported NIfTI datatype: " + niftiHeader.datatypeCode);

      const sliceSize = cols * rows;
      const is4D = timeFrames > 1;

      // For 4D cine: extract all time frames at the middle slice (for EF calc)
      // For 3D: extract all spatial slices (for Z-scrubber)
      const allSlicesB64: string[] = [];

      if (is4D) {
        // Cine MRI: return all time frames at the middle Z-slice
        const midSlice = Math.floor(slices / 2);
        for (let t = 0; t < timeFrames; t++) {
          const offset = (t * slices * sliceSize) + (midSlice * sliceSize);
          const raw = Array.from({ length: sliceSize }, (_, i) => (typedData as any)[offset + i]);
          const norm = normalizeToUint8(raw);
          allSlicesB64.push(Buffer.from(norm).toString('base64'));
        }
        return NextResponse.json({
          cols, rows,
          sliceCount: timeFrames,
          mode: '4d_cine',
          slices: allSlicesB64
        });
      } else {
        // 3D volume: return all Z-axis slices
        for (let s = 0; s < slices; s++) {
          const raw = Array.from({ length: sliceSize }, (_, i) => (typedData as any)[s * sliceSize + i]);
          const norm = normalizeToUint8(raw);
          allSlicesB64.push(Buffer.from(norm).toString('base64'));
        }
        return NextResponse.json({
          cols, rows,
          sliceCount: slices,
          mode: '3d_volume',
          slices: allSlicesB64
        });
      }
    }

    // ═══════════════════════════════════════
    // DICOM path: single 2D slice
    // ═══════════════════════════════════════
    const dataView = new DataView(buffer);
    const image = daikon.Series.parseImage(dataView);
    if (!image) throw new Error("Could not parse DICOM image.");

    const cols = image.getCols();
    const rows = image.getRows();
    const pixels = image.getInterpretedData();
    const normalized = normalizeToUint8(pixels);
    const base64Str = Buffer.from(normalized).toString('base64');

    return NextResponse.json({
      cols, rows,
      sliceCount: 1,
      mode: 'dicom',
      slices: [base64Str]
    });

  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
