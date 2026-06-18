#!/usr/bin/env python3
"""
UV-unwrap a vertex-colored PLY mesh and bake colors to a texture atlas.
Outputs: <stem>_textured.obj  +  <stem>_textured.mtl  +  <stem>_textured.png

Usage:
  python bake_texture.py --mesh /path/to/mesh.ply [--out_dir /path] [--tex_size 4096]
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from PIL import Image


def _dilate_texture(tex, iters=16):
    """Push filled texel colors into neighbouring unfilled (black) pixels to hide UV seams."""
    try:
        from scipy.ndimage import grey_dilation, binary_dilation
        mask = tex.max(axis=2) > 0
        for _ in range(iters):
            dilated = np.stack([grey_dilation(tex[:, :, c], size=3) for c in range(3)], axis=2)
            grown = binary_dilation(mask)
            border = grown & ~mask
            tex[border] = dilated[border]
            mask = grown
        return tex
    except ImportError:
        # Simple 4-neighbour fallback (no scipy)
        mask = tex.max(axis=2) > 0
        for _ in range(iters):
            new_tex = tex.copy()
            new_mask = mask.copy()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                s_tex  = np.roll(np.roll(tex,  dy, axis=0), dx, axis=1)
                s_mask = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
                fill = ~new_mask & s_mask
                new_tex[fill] = s_tex[fill]
                new_mask |= fill
            tex, mask = new_tex, new_mask
        return tex


def _ensure_xatlas():
    try:
        import xatlas
        return xatlas
    except ImportError:
        import subprocess
        print("Installing xatlas...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xatlas", "-q"])
        import xatlas
        return xatlas


def _bary(pts, a, b, c):
    """Vectorised barycentric coords of pts (Nx2) w.r.t. triangle (a,b,c) each shape (2,)."""
    v0 = b - a
    v1 = c - a
    v2 = pts - a
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    d20 = v2 @ v0
    d21 = v2 @ v1
    inv = 1.0 / (d00 * d11 - d01 * d01 + 1e-10)
    bv = (d11 * d20 - d01 * d21) * inv
    bw = (d00 * d21 - d01 * d20) * inv
    bu = 1.0 - bv - bw
    return bu, bv, bw


def bake(ply_path, out_dir, tex_size=4096):
    xatlas = _ensure_xatlas()

    stem = os.path.splitext(os.path.basename(ply_path))[0] + "_textured"

    print(f"[bake] Loading {ply_path}")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()

    verts   = np.asarray(mesh.vertices,       dtype=np.float32)
    tris    = np.asarray(mesh.triangles,      dtype=np.uint32)
    colors  = np.asarray(mesh.vertex_colors,  dtype=np.float32)  # 0-1
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    print(f"[bake] UV unwrapping {len(verts):,} verts / {len(tris):,} tris ...")
    vmapping, indices, uvs = xatlas.parametrize(verts, tris, normals)

    new_verts  = verts[vmapping]
    new_colors = colors[vmapping]  # per unique UV vertex

    # ── rasterise vertex colors into texture ──────────────────────────────────
    print(f"[bake] Rasterising {len(indices):,} triangles → {tex_size}x{tex_size} ...")
    tex = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)

    uv_px = uvs * (tex_size - 1)   # pixel-space UV coords, shape (N, 2)

    for i, tri in enumerate(indices):
        if i % 100_000 == 0 and i:
            print(f"[bake]   {i:,} / {len(indices):,}")

        p0, p1, p2 = uv_px[tri[0]], uv_px[tri[1]], uv_px[tri[2]]
        c0, c1, c2 = new_colors[tri[0]], new_colors[tri[1]], new_colors[tri[2]]

        x0 = max(int(min(p0[0], p1[0], p2[0])), 0)
        x1 = min(int(max(p0[0], p1[0], p2[0])) + 1, tex_size - 1)
        y0 = max(int(min(p0[1], p1[1], p2[1])), 0)
        y1 = min(int(max(p0[1], p1[1], p2[1])) + 1, tex_size - 1)
        if x0 > x1 or y0 > y1:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

        bu, bv, bw = _bary(pts, p0, p1, p2)
        mask = (bu >= 0) & (bv >= 0) & (bw >= 0)
        if not mask.any():
            continue

        col = bu[mask, None] * c0 + bv[mask, None] * c1 + bw[mask, None] * c2
        py_f = pts[mask, 1].astype(int)
        px_f = pts[mask, 0].astype(int)
        tex[py_f, px_f] = (col * 255).clip(0, 255).astype(np.uint8)

    # ── save texture ──────────────────────────────────────────────────────────
    tex_name = stem + ".png"
    tex_path = os.path.join(out_dir, tex_name)
    Image.fromarray(tex).save(tex_path)
    print(f"[bake] Texture  → {tex_path}")

    # ── write MTL ─────────────────────────────────────────────────────────────
    mtl_name = stem + ".mtl"
    mtl_path = os.path.join(out_dir, mtl_name)
    with open(mtl_path, "w") as f:
        f.write("newmtl mat0\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {tex_name}\n")
    print(f"[bake] MTL      → {mtl_path}")

    # ── write OBJ ─────────────────────────────────────────────────────────────
    obj_name = stem + ".obj"
    obj_path = os.path.join(out_dir, obj_name)
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write(f"o {stem}\n")
        for v in new_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")  # flip Y for OBJ convention
        f.write("usemtl mat0\n")
        for t in indices:
            i0, i1, i2 = t[0] + 1, t[1] + 1, t[2] + 1
            f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")
    print(f"[bake] OBJ      → {obj_path}")

    return obj_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bake vertex colors to UV texture atlas")
    ap.add_argument("--mesh",     required=True,       help="Input PLY path (with vertex colors)")
    ap.add_argument("--out_dir",  default=None,        help="Output directory (default: same as mesh)")
    ap.add_argument("--tex_size", type=int, default=4096, help="Texture resolution (default 4096)")
    args = ap.parse_args()

    out = args.out_dir or os.path.dirname(os.path.abspath(args.mesh))
    bake(args.mesh, out, args.tex_size)
