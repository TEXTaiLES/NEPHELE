import requests
import os
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
HESTIA_API_URL = f"{os.getenv('HESTIA_API_URL', 'http://api.textailes.athenarc.gr')}/robot-images"
API_KEY = os.getenv('HESTIA_API_KEY')

# Based on run_pipeline.sh, SAM2 expects: .../SAM2/data/input/<DATASET_NAME>
# Adjust RELATIVE_SAM2_PATH to point to your SAM2 folder relative to this script
RELATIVE_SAM2_PATH = "./SAMplify_SuGaR/SAM2/data/input"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
def get_camera_from_pipeline(pipeline, split: str, index: int, device: str):
    """
    Nerfstudio versions differ:
    - Some datasets return item["cameras"]
    - Others return only {"image_idx","image"} and cameras live in dataparser_outputs.cameras
    This function supports both.
    """
    dm = getattr(pipeline, "datamanager", None)
    if dm is None:
        raise RuntimeError("Pipeline has no datamanager; nerfstudio version mismatch?")

    # 1) Try dataparser outputs cameras (most robust)
    dpo = None
    if split == "train":
        for attr in ["train_dataparser_outputs", "train_dataparser_output", "train_dataparser"]:
            if hasattr(dm, attr):
                dpo = getattr(dm, attr)
                break
        if dpo is None and hasattr(dm, "train_dataparser_outputs"):
            dpo = dm.train_dataparser_outputs
    else:
        for attr in ["eval_dataparser_outputs", "val_dataparser_outputs", "test_dataparser_outputs"]:
            if hasattr(dm, attr):
                dpo = getattr(dm, attr)
                break
        # some versions keep eval outputs only
        if dpo is None and hasattr(dm, "eval_dataparser_outputs"):
            dpo = dm.eval_dataparser_outputs

    if dpo is not None and hasattr(dpo, "cameras"):
        cams = dpo.cameras  # Cameras object with batch dimension = num images
        # Some datasets are wrapped; index is within that split ordering.
        cam = cams[index : index + 1].to(device)
        return cam

    # 2) Fallback: try dataset item
    dataset = None
    if split == "train" and hasattr(dm, "train_dataset"):
        dataset = dm.train_dataset
    elif split in ["val", "test"] and hasattr(dm, "eval_dataset"):
        dataset = dm.eval_dataset

    if dataset is None:
        raise RuntimeError(f"Couldn't find dataset/dataparser_outputs for split={split}.")

    item = dataset[index]
    if "cameras" in item:
        return item["cameras"].to(device)
    if "camera" in item:
        return item["camera"].to(device)

    # 3) Fallback using image_idx -> cameras
    if "image_idx" in item:
        img_i = int(item["image_idx"])
        # try again: some versions store cameras on dataset itself
        if hasattr(dataset, "cameras"):
            cam = dataset.cameras[img_i : img_i + 1].to(device)
            return cam

    raise RuntimeError(f"Dataset item has no camera key. Keys={list(item.keys())}")


def download_scan(scan_id):
    if not scan_id:
        logger.error("Error: A scan_id is required.")
        logger.info("Usage: python download_robot_images.py <SCAN_ID>")
        return

    if not API_KEY:
        logger.error("API_KEY not found in .env file.")
        return

    headers = {"Authorization": f"Bearer {API_KEY}"}

    # 1. Setup Directory
    dataset_name = f"scan_{scan_id}"
    output_dir = os.path.join(RELATIVE_SAM2_PATH, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Target Directory: {output_dir}")
    logger.info(f"Fetching all images for Scan ID: {scan_id}")

    # 2. Pagination Loop
    page = 1
    per_page = 100
    total_downloaded = 0

    while True:
        logger.info(f"   Fetching page {page}...")

        try:
            params = {
                "scan_id": scan_id,
                "page": page,
                "per_page": per_page
            }
            response = requests.get(HESTIA_API_URL, headers=headers, params=params)

            if response.status_code != 200:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                break

            batch_data = response.json()

            # Stop if no more data returned
            if not batch_data:
                break

            # 3. Process Batch
            for record in batch_data:
                image_url = record.get('public_url')
                filename = record.get('filename')

                if not image_url or not filename:
                    continue

                save_path = os.path.join(output_dir, filename)

                # Skip if already downloaded
                if os.path.exists(save_path):
                    continue

                # Download with basic retry
                try:
                    img_resp = requests.get(image_url, timeout=10)
                    if img_resp.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(img_resp.content)
                        total_downloaded += 1
                        print(f"   Downloaded: {filename}", end='\r')
                    else:
                        logger.warning(f"   Failed to download {filename}: Status {img_resp.status_code}")
                except Exception as e:
                    logger.warning(f"   Exception downloading {filename}: {e}")

            # Prepare next page
            page += 1

        except Exception as e:
            logger.error(f"Critical Error: {e}")
            break

    print("")
    if total_downloaded > 0:
        logger.info(f"Batch Complete. {total_downloaded} new images saved.")
        logger.info(f"To run reconstruction: ./run_pipeline.sh {dataset_name}")
    else:
        logger.info("No new images found or downloaded.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_id = sys.argv[1]
        download_scan(target_id)
    else:
        print("Please provide a Scan ID.")
        print("Example: python download_robot_images.py 123e4567-e89b-...")