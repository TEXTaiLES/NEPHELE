#!/usr/bin/env python3
"""
Convert undistorted masked images to RGBA PNGs with alpha channel.
Background pixels (near-black from SAM2 masking) get alpha=0.
Foreground pixels get alpha=255.
Handles both JPEG and PNG source images.
Also patches images.bin to reference .png filenames if needed.
"""
import os
import sys
import glob
import shutil
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


def convert_images(images_dir, images_bin_path=None, black_threshold=15, erosion_px=2):
    # Accept both JPG and PNG sources
    img_files = sorted(
        glob.glob(os.path.join(images_dir, '*.jpg')) +
        glob.glob(os.path.join(images_dir, '*.jpeg')) +
        glob.glob(os.path.join(images_dir, '*.png'))
    )
    if not img_files:
        print(f"  No image files found in {images_dir}, skipping.")
        return

    # Detect source extension (may be mixed, but usually uniform)
    exts = set(os.path.splitext(f)[1].lower() for f in img_files)
    print(f"  Found {len(img_files)} image(s) with extension(s): {exts}")

    converted = 0
    skipped = 0
    for src_path in img_files:
        img = Image.open(src_path)
        src_ext = os.path.splitext(src_path)[1].lower()

        # If already RGBA PNG, check if it has a real alpha channel (not all 255)
        if img.mode == 'RGBA' and src_ext == '.png':
            alpha = np.array(img)[:, :, 3]
            if alpha.min() < 255:
                skipped += 1
                continue  # Already properly alpha-masked

        # Build RGBA with black-background masking
        rgb = np.array(img.convert('RGB'))
        fg_mask = rgb.max(axis=2) > black_threshold
        if erosion_px > 0:
            struct = np.ones((erosion_px * 2 + 1, erosion_px * 2 + 1), dtype=bool)
            fg_mask = binary_erosion(fg_mask, structure=struct)
        alpha = fg_mask.astype(np.uint8) * 255
        rgba = np.dstack([rgb, alpha])

        png_path = os.path.splitext(src_path)[0] + '.png'
        Image.fromarray(rgba, 'RGBA').save(png_path)

        # Remove original only if it was a JPEG (PNG is overwritten in-place)
        if src_ext in ('.jpg', '.jpeg'):
            os.remove(src_path)

        fg_pct = 100.0 * (alpha > 0).mean()
        print(f"    {os.path.basename(png_path)}  foreground: {fg_pct:.1f}%")
        converted += 1

    print(f"  Converted: {converted}, already RGBA: {skipped}")

    # Patch images.bin: replace .jpg\0 -> .png\0 if needed
    if images_bin_path and os.path.exists(images_bin_path):
        with open(images_bin_path, 'rb') as f:
            data = f.read()

        if b'.jpg\x00' in data or b'.jpeg\x00' in data:
            shutil.copy2(images_bin_path, images_bin_path + '.orig')
            patched = data.replace(b'.jpg\x00', b'.png\x00').replace(b'.jpeg\x00', b'.png\x00')
            replacements = data.count(b'.jpg\x00') + data.count(b'.jpeg\x00')
            print(f"  Patched images.bin: {replacements} filename(s) updated -> .png")
            with open(images_bin_path, 'wb') as f:
                f.write(patched)
        else:
            print("  images.bin already references .png — no patching needed")
    else:
        print("  [!] images.bin not found or not provided — skipping patch")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <images_dir> [images.bin path] [erosion_px]")
        sys.exit(1)

    images_dir = sys.argv[1]
    images_bin = sys.argv[2] if len(sys.argv) > 2 else None
    erosion_px = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    convert_images(images_dir, images_bin, erosion_px=erosion_px)

