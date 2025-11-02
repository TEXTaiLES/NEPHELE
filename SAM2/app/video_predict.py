import os, glob, json, cv2, torch, numpy as np
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
    if all(_is_indexed_name(p) for p in files):
        return src_dir
    parent = os.path.dirname(os.path.abspath(src_dir.rstrip("/")))
    base   = os.path.basename(src_dir.rstrip("/"))
    dst    = os.path.join(parent, base + suffix)
    os.makedirs(dst, exist_ok=True)
    existing = _gather_frames(dst)
    if len(existing) == len(files) and all(
        os.path.exists(os.path.join(dst, f"{i:06d}{os.path.splitext(files[i])[1].lower()}"))
        for i in range(len(files))
    ):
        return dst
    for i, f in enumerate(files):
        ext = os.path.splitext(f)[1].lower()
        if ext not in ACCEPT_EXTS: continue
        newp = os.path.join(dst, f"{i:06d}{ext}")
        if os.path.exists(newp): continue
        try: os.symlink(os.path.abspath(f), newp)
        except OSError: shutil.copy2(f, newp)
    return dst

def to_u8_mask(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze()
        if x.is_floating_point(): x = x > 0.5
        x = x.to(torch.uint8).numpy()
    else:
        x = np.asarray(x).squeeze()
        if x.dtype.kind == "f": x = x > 0.5
        x = x.astype(np.uint8)
    return x * 255

def save_color_cutout(orig_img_path, mask_u8, out_path):
    img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(orig_img_path)
    m = mask_u8[..., 0] if mask_u8.ndim == 3 else mask_u8
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    cutout = cv2.bitwise_and(img, img, mask=m)
    cv2.imwrite(out_path, cutout)

# ===================== MAIN =====================
# 1) Index frames
if os.path.isdir(INPUT) and AUTO_INDEX == "1":
    INPUT = ensure_indexed(INPUT, INDEX_SUFFIX)

# 2) Paths
inp_base = os.path.basename(INPUT.rstrip("/"))
out_name = inp_base if os.path.isdir(INPUT) else os.path.splitext(inp_base)[0]
OUT_DIR = os.path.join(OUT_ROOT, out_name)
OUT_MASKED_DIR = os.path.join(OUT_ROOT, f"{out_name}_masked")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_MASKED_DIR, exist_ok=True)
PROMPTS_JSON = os.path.join(OUT_DIR, "prompts.json")

# 3) Require prompts.json (from the Flask picker)
if not os.path.isfile(PROMPTS_JSON):
    raise FileNotFoundError(
        f"prompts.json not found at {PROMPTS_JSON}. "
        f"Run the web point picker first and press 'Save prompts.json'."
    )

with open(PROMPTS_JSON, "r") as f:
    J = json.load(f)

points    = np.array(J["points"], dtype=np.float32)
labels    = np.array(J["labels"], dtype=np.int32)
FRAME_IDX = int(J["frame_idx"])
OBJ_ID    = int(J["obj_id"])

# 4) Validate frames + name mapping
frame_paths = _gather_frames(INPUT)
if not frame_paths:
    raise FileNotFoundError(f"No frames found in {INPUT} (accepted: {', '.join(ACCEPT_EXTS)})")
idx_to_orig_name = [os.path.basename(os.path.realpath(p)) for p in frame_paths]

# 5) SAM2
pred = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large").to("cuda")
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
dtype = torch.bfloat16 if use_bf16 else torch.float16

with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
    state = pred.init_state(INPUT)

    # Prompted frame
    frame_idx, obj_ids, masks = pred.add_new_points_or_box(
        state, FRAME_IDX, OBJ_ID, points=points, labels=labels
    )

    def save_set(frame_idx, obj_ids, masks):
        base_name = idx_to_orig_name[frame_idx]
        stem, orig_ext = os.path.splitext(base_name)
        single_obj = len(obj_ids) == 1
        orig_img_path = os.path.realpath(frame_paths[frame_idx])
        ext_for_save = orig_ext if orig_ext else ".jpg"
        for k, oid in enumerate(obj_ids):
            suffix = "" if single_obj else f"_obj{oid}"
            out_name = f"{stem}{suffix}{ext_for_save}"
            mask_u8  = to_u8_mask(masks[k])
            cv2.imwrite(os.path.join(OUT_DIR, out_name), mask_u8)
            save_color_cutout(orig_img_path, mask_u8, os.path.join(OUT_MASKED_DIR, out_name))

    save_set(frame_idx, obj_ids, masks)

    # Propagate to remaining frames
    for frame_idx, obj_ids, masks in pred.propagate_in_video(state):
        base_name = idx_to_orig_name[frame_idx] if idx_to_orig_name else f"{frame_idx:06d}"
        stem, orig_ext = os.path.splitext(base_name)
        single_obj = len(obj_ids) == 1
        orig_img_path = os.path.realpath(frame_paths[frame_idx]) if frame_paths else None
        ext_for_save = orig_ext if orig_ext else ".jpg"

        for k, oid in enumerate(obj_ids):
            suffix = "" if single_obj else f"_obj{oid}"
            out_name = f"{stem}{suffix}{ext_for_save}"

            mask_path = os.path.join(OUT_DIR, out_name)
            mask_u8   = to_u8_mask(masks[k])
            cv2.imwrite(mask_path, mask_u8)

            if orig_img_path:
                cutout_path = os.path.join(OUT_MASKED_DIR, out_name)
                save_color_cutout(orig_img_path, mask_u8, cutout_path)
