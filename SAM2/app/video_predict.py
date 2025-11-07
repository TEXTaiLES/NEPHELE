#!/usr/bin/env python3
import os, glob, json, cv2, torch, numpy as np
import argparse

from sam2.sam2_video_predictor import SAM2VideoPredictor

# ===================== CONFIG =====================
DATASET_NAME = os.environ.get("DATASET_NAME", "").strip()
_default_input = f"/data/in/{DATASET_NAME}" if DATASET_NAME else "/data/in"
INPUT = os.environ.get("INPUT", _default_input)
OUT_ROOT = os.environ.get("OUT", "/data/out")

AUTO_INDEX   = os.environ.get("AUTO_INDEX", "1")
INDEX_SUFFIX = os.environ.get("INDEX_SUFFIX", "_indexed")

QUIET = os.environ.get("QUIET", "1")  # 0 to see logs

ACCEPT_EXTS = (".jpg", ".jpeg", ".png")

if QUIET == "1":
    import sys, warnings
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore")
    try:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    except Exception:
        pass


# ===================== HELPERS =====================
def _gather_frames(d):
    files = []
    for ext in ACCEPT_EXTS:
        files += glob.glob(os.path.join(d, f"*{ext}"))
        files += glob.glob(os.path.join(d, f"*{ext.upper()}"))
    return sorted(set(os.path.abspath(p) for p in files))


def _is_indexed_name(p):
    n = os.path.splitext(os.path.basename(p))[0]
    return len(n) == 6 and n.isdigit()


def ensure_indexed(src_dir, suffix="_indexed"):
    import shutil
    if not os.path.isdir(src_dir):
        raise NotADirectoryError(f"Not a directory: {src_dir}")
    files = _gather_frames(src_dir)
    if not files:
        raise FileNotFoundError(f"No frames in {src_dir} (accepted: {', '.join(ACCEPT_EXTS)})")

    parent = os.path.dirname(os.path.abspath(src_dir.rstrip("/")))
    base   = os.path.basename(src_dir.rstrip("/"))
    dst    = os.path.join(parent, base + suffix)
    os.makedirs(dst, exist_ok=True)

    # Rebuild the indexed set every time (simple & safe)
    for f in os.listdir(dst):
        os.remove(os.path.join(dst, f))

    for i, f in enumerate(files):
        ext = os.path.splitext(f)[1].lower()
        if ext not in ACCEPT_EXTS:
            continue
        newp = os.path.join(dst, f"{i:06d}{ext}")
        shutil.copy2(f, newp)

    return dst



def to_u8_mask(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze()
        if x.is_floating_point():
            x = x > 0.5
        x = x.to(torch.uint8).numpy()
    else:
        x = np.asarray(x).squeeze()
        if x.dtype.kind == "f":
            x = x > 0.5
        x = x.astype(np.uint8)
    return x * 255


def save_color_cutout(orig_img_path, mask_u8, out_path):
    img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(orig_img_path)
    m = mask_u8[..., 0] if mask_u8.ndim == 3 else mask_u8
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(
            m, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    cutout = cv2.bitwise_and(img, img, mask=m)
    cv2.imwrite(out_path, cutout)

def save_overlay_preview(
    orig_img_path,
    mask_u8,
    out_path,
    dim_alpha: float = 0.6,
    border_color=None,   # None => no border
):
    """
    Preview visualization:
      - Keep the masked region in original colors.
      - Darken everything outside the mask by `dim_alpha`.
      - Optionally draw a thin colored border around the mask
        if `border_color` is not None.

    dim_alpha ~ 0.5–0.7 works well (0.6 = 60% darker background).
    """
    img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(orig_img_path)

    # Ensure mask matches image resolution
    m = mask_u8[..., 0] if mask_u8.ndim == 3 else mask_u8
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(
            m, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Binary 0/255 mask
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    # Boolean mask (0/1)
    mask = (m > 0).astype(np.uint8)

    # 3-channel version of the mask
    mask3 = cv2.merge([mask, mask, mask])  # (H, W, 3)

    # Darkened background
    img_f = img.astype(np.float32)
    dim_factor = 1.0 - float(dim_alpha)     # e.g. 0.4 if dim_alpha=0.6
    dimmed = np.clip(img_f * dim_factor, 0, 255).astype(np.uint8)

    # Keep original where mask==1, dimmed where mask==0
    out = np.where(mask3 == 1, img, dimmed)

    # Optional border
    if border_color is not None:
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        border = ((dilated - mask) > 0).astype(np.uint8)
        border3 = cv2.merge([border, border, border])
        border_color_arr = np.array(border_color, dtype=np.uint8).reshape(1, 1, 3)
        out = np.where(border3 == 1, border_color_arr, out)

    cv2.imwrite(out_path, out)



# ===================== CORE LOGIC =====================

def run_sam2(preview=False, preview_num_frames=6, preview_out=None):
    global INPUT

    # 1) Index frames if needed
    if os.path.isdir(INPUT) and AUTO_INDEX == "1":
        INPUT = ensure_indexed(INPUT, INDEX_SUFFIX)

    # 2) Paths
    inp_base = os.path.basename(INPUT.rstrip("/"))
    out_name = inp_base if os.path.isdir(INPUT) else os.path.splitext(inp_base)[0]

    OUT_DIR = os.path.join(OUT_ROOT, out_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Where the visual previews / cutouts go
    if preview and preview_out is not None:
        OUT_MASKED_DIR = preview_out          # for preview mode
    else:
        OUT_MASKED_DIR = os.path.join(OUT_ROOT, f"{out_name}_masked")

    os.makedirs(OUT_MASKED_DIR, exist_ok=True)

    PROMPTS_JSON = os.path.join(OUT_DIR, "prompts.json")
    if not os.path.isfile(PROMPTS_JSON):
        raise FileNotFoundError(
            f"prompts.json not found at {PROMPTS_JSON}. "
            f"Run the web point picker first and press 'Save'."
        )

    with open(PROMPTS_JSON, "r") as f:
        J = json.load(f)

    points    = np.array(J["points"], dtype=np.float32)
    labels    = np.array(J["labels"], dtype=np.int32)
    FRAME_IDX = int(J["frame_idx"])   # annotated frame
    OBJ_ID    = int(J["obj_id"])

    # 4) Validate frames + name mapping
    frame_paths = _gather_frames(INPUT)
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in {INPUT} (accepted: {', '.join(ACCEPT_EXTS)})"
        )
    idx_to_orig_name = [os.path.basename(os.path.realpath(p)) for p in frame_paths]
    num_total_frames = len(frame_paths)

    # 5) SAM2 predictor
    pred = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large").to("cuda")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Are we in preview mode?
    is_preview_mode = bool(preview and preview_out is not None)

    def save_set(frame_idx, obj_ids, masks):
        """
        Save masks + visualization for a given frame.
        """
        base_name = idx_to_orig_name[frame_idx]
        stem, orig_ext = os.path.splitext(base_name)
        single_obj = len(obj_ids) == 1
        orig_img_path = os.path.realpath(frame_paths[frame_idx])
        ext_for_save = orig_ext if orig_ext else ".jpg"

        for k, oid in enumerate(obj_ids):
            suffix   = "" if single_obj else f"_obj{oid}"
            out_name = f"{stem}{suffix}{ext_for_save}"

            # 1) binary mask -> OUT_DIR
            mask_u8   = to_u8_mask(masks[k])
            mask_path = os.path.join(OUT_DIR, out_name)
            cv2.imwrite(mask_path, mask_u8)

            # 2) visualization -> OUT_MASKED_DIR
            vis_path = os.path.join(OUT_MASKED_DIR, out_name)
            if is_preview_mode:
                # RGB object, dimmed background, no colored border
                save_overlay_preview(
                    orig_img_path,
                    mask_u8,
                    vis_path,
                    dim_alpha=0.9,    # tweak if you want more/less dimming
                    border_color=None # <— this removes the green outline
                )
            else:
                save_color_cutout(orig_img_path, mask_u8, vis_path)



    # ===================== SAM2 inference =====================
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        state = pred.init_state(INPUT)

        # First, the annotated frame (always included in preview)
        frame_idx, obj_ids, masks = pred.add_new_points_or_box(
            state, FRAME_IDX, OBJ_ID, points=points, labels=labels
        )
        save_set(frame_idx, obj_ids, masks)

        if preview:
            import random

            # We want: 1 annotated + N random extra frames
            extra_needed = max(0, min(preview_num_frames - 1, num_total_frames - 1))

            # candidate indices except the annotated frame
            candidates = [i for i in range(num_total_frames) if i != FRAME_IDX]
            chosen_extra = set(random.sample(candidates, extra_needed)) if extra_needed > 0 else set()

            saved = 1  # already saved annotated frame

            for frame_idx, obj_ids, masks in pred.propagate_in_video(state):
                if frame_idx in chosen_extra:
                    save_set(frame_idx, obj_ids, masks)
                    saved += 1
                    if saved >= max(1, preview_num_frames):
                        break
        else:
            # full propagation for the pipeline
            for frame_idx, obj_ids, masks in pred.propagate_in_video(state):
                save_set(frame_idx, obj_ids, masks)




# ===================== ENTRY POINT =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Run in preview mode (few frames) and write cutouts to --preview-out"
    )
    parser.add_argument(
        "--preview-num-frames",
        type=int,
        default=6,
        help="Number of frames (including the prompted one) to save in preview mode"
    )
    parser.add_argument(
        "--preview-out",
        type=str,
        default=None,
        help="Directory for preview cutouts (required for --preview)"
    )

    # ignore unknown args (caller may pass more)
    args, _ = parser.parse_known_args()

    run_sam2(
        preview=args.preview,
        preview_num_frames=args.preview_num_frames,
        preview_out=args.preview_out,
    )
