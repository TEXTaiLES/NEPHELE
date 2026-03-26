#!/usr/bin/env python3
"""
Post-process a SuGaR OBJ mesh by filtering faces directly in the OBJ file.
Preserves UV coordinates, normals, and materials exactly.

Steps:
  1. Remove small disconnected components (geometric connectivity)
  2. Crop bottom skirt (faces whose lowest vertex is below a Y percentile)
"""
import sys
import os
import shutil
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def parse_obj(path):
    """Parse OBJ into vertices, faces (as raw index tuples), and other lines."""
    vertices = []   # (x, y, z)
    faces = []      # list of [ (vi, vti, vni), ... ] — 1-based indices as strings
    other_lines = []  # all non-v/non-f lines in order, with position markers

    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            parts = line.split()
            if not parts:
                other_lines.append(line)
                continue
            if parts[0] == 'v':
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == 'f':
                # each token is v, v/vt, or v/vt/vn
                face = []
                for tok in parts[1:]:
                    face.append(tok)
                faces.append(face)
            else:
                other_lines.append(line)

    return np.array(vertices), faces, other_lines


def face_geo_verts(face_tokens):
    """Return 0-based geometric vertex indices for a face."""
    indices = []
    for tok in face_tokens:
        v = int(tok.split('/')[0])
        indices.append(v - 1 if v > 0 else v)  # handle negative indices
    return indices


def build_components(vertices, all_faces):
    """Build connected components using geometric vertex positions."""
    rounded = np.round(vertices, decimals=5)
    _, geo_ids = np.unique(rounded, axis=0, return_inverse=True)

    n_clusters = int(geo_ids.max()) + 1
    face_geo = [geo_ids[face_geo_verts(f)] for f in all_faces]

    rows, cols = [], []
    for fg in face_geo:
        for i in range(len(fg)):
            rows.append(fg[i]);  cols.append(fg[(i+1) % len(fg)])
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_clusters, n_clusters))
    adj = (adj + adj.T).astype(bool)

    _, labels = connected_components(adj, directed=False)
    face_comp = np.array([labels[fg[0]] for fg in face_geo])
    return face_comp


def write_obj(path, vertices, kept_faces, other_lines, orig_path):
    """Write filtered OBJ preserving all vertex/vt/vn lines and only kept faces."""
    # Find which geometric vertex indices are referenced
    used_v = set()
    for face in kept_faces:
        for tok in face:
            vi = int(tok.split('/')[0])
            used_v.add(vi - 1 if vi > 0 else vi)

    # Remap vertex indices (we keep ALL v/vt/vn lines for simplicity —
    # unused ones don't affect rendering but keep file structure intact)
    with open(orig_path) as fin, open(path, 'w') as fout:
        face_iter = iter(kept_faces)
        for line in fin:
            stripped = line.rstrip('\n')
            parts = stripped.split()
            if not parts:
                fout.write(line)
            elif parts[0] == 'f':
                # skip — we'll write kept faces at the end via face_iter
                pass
            elif parts[0] == 'v' and len(parts) == 4:
                fout.write(line)  # keep all geometry vertices
            else:
                fout.write(line)  # vt, vn, mtllib, usemtl, comments, etc.

        # Write kept faces
        for face in kept_faces:
            fout.write('f ' + ' '.join(face) + '\n')


def cleanup_mesh(obj_path, min_ratio=0.05, bottom_crop_percentile=None, crop_axis=1):
    vertices, faces, _ = parse_obj(obj_path)
    if len(faces) == 0:
        print("  No faces found.")
        return

    keep_mask = np.ones(len(faces), dtype=bool)

    # --- Step 1: remove small disconnected components ---
    face_comp = build_components(vertices, faces)
    comp_sizes = np.bincount(face_comp)
    largest = comp_sizes.max()
    n_comp = len(comp_sizes)
    print(f"  Components: {n_comp}, largest: {largest} / {len(faces)} faces")

    # Identify the "main" component as the one whose centroid is closest to the
    # median centroid of all faces. This is more robust than using the largest
    # component when a large floater artifact outweighs the actual object.
    face_centroids_all = np.array([
        vertices[face_geo_verts(f)].mean(axis=0) for f in faces
    ])
    median_centroid = np.median(face_centroids_all, axis=0)
    significant = np.where(comp_sizes >= 0.001 * len(faces))[0]
    main_comp = min(significant, key=lambda c: np.linalg.norm(
        face_centroids_all[face_comp == c].mean(axis=0) - median_centroid
    ))
    main_size = comp_sizes[main_comp]
    print(f"  Main component: {main_comp} ({main_size} faces, "
          f"centroid dist from median={np.linalg.norm(face_centroids_all[face_comp==main_comp].mean(0)-median_centroid):.3f})")

    keep_comps = set(np.where(comp_sizes >= min_ratio * main_size)[0])
    comp_mask = np.array([face_comp[i] in keep_comps for i in range(len(faces))])
    n_comp_removed = int((~comp_mask).sum())
    keep_mask &= comp_mask
    if n_comp_removed:
        print(f"  Removed {n_comp_removed} faces ({n_comp - len(keep_comps)} small components)")

    # --- Step 2: bottom crop ---
    if bottom_crop_percentile is not None:
        axis_vals = vertices[:, crop_axis]
        cutoff = np.percentile(axis_vals, bottom_crop_percentile)
        axis_name = ['X', 'Y', 'Z'][crop_axis]
        # Remove faces whose LOWEST vertex (on crop axis) is below the cutoff
        face_min_ax = np.array([
            min(vertices[vi, crop_axis] for vi in face_geo_verts(f))
            for f in faces
        ])
        base_mask = face_min_ax >= cutoff
        n_base_removed = int((~base_mask & keep_mask).sum())
        keep_mask &= base_mask
        if n_base_removed:
            print(f"  Removed {n_base_removed} faces below {axis_name}={cutoff:.4f} (bottom {bottom_crop_percentile:.0f}%)")

    # --- Step 3: horizontal outlier crop (IQR-based, axes other than crop_axis) ---
    # Removes faces whose centroid is far from the object in the horizontal plane.
    # Catches floor/wall planes that are connected to the object (missed by component filter).
    face_centroids = np.array([
        vertices[face_geo_verts(f)].mean(axis=0) for f in faces
    ])
    outlier_mask = np.ones(len(faces), dtype=bool)
    for ax in range(3):
        if ax == crop_axis:
            continue  # skip the vertical axis (already handled by bottom crop)
        vals = face_centroids[keep_mask, ax]
        if len(vals) < 4:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-6:
            continue
        lo = q1 - 3.0 * iqr
        hi = q3 + 3.0 * iqr
        ax_mask = (face_centroids[:, ax] >= lo) & (face_centroids[:, ax] <= hi)
        n_outliers = int((~ax_mask & keep_mask).sum())
        if n_outliers:
            axis_name = ['X', 'Y', 'Z'][ax]
            print(f"  Removed {n_outliers} faces outside {axis_name}=[{lo:.3f}, {hi:.3f}] (IQR outliers)")
        outlier_mask &= ax_mask
    keep_mask &= outlier_mask

    n_total_removed = int((~keep_mask).sum())
    if n_total_removed == 0:
        print("  Nothing to remove.")
        return

    kept_faces = [f for f, k in zip(faces, keep_mask) if k]
    print(f"  Keeping {len(kept_faces)} / {len(faces)} faces")

    backup = os.path.join(os.path.dirname(obj_path),
                          os.path.basename(obj_path).replace('.obj', '_before_cleanup.obj'))
    shutil.copy2(obj_path, backup)

    write_obj(obj_path, vertices, kept_faces, None, backup)
    print(f"  Saved: {obj_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <mesh.obj> [min_ratio=0.05] [bottom_crop_pct]")
        sys.exit(1)

    path = sys.argv[1]
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    crop = float(sys.argv[3]) if len(sys.argv) > 3 else None
    axis = int(sys.argv[4]) if len(sys.argv) > 4 else 1  # 0=X, 1=Y, 2=Z
    cleanup_mesh(path, min_ratio=ratio, bottom_crop_percentile=crop, crop_axis=axis)
