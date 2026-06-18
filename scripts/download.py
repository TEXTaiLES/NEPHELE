import requests
import os
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
HESTIA_API_URL = f"{os.getenv('HESTIA_API_URL', 'http://api.textailes.athenarc.gr')}/robot-images"
API_KEY = os.getenv('HESTIA_API_KEY')

# SAM2 expects input images at: <repo>/SAM2/data/input/<DATASET_NAME>
RELATIVE_SAM2_PATH = "../SAM2/data/input"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

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

                # Download with proxy authentication and streaming
                try:
                    img_resp = requests.get(image_url, headers=headers, stream=True, timeout=10)
                    if img_resp.status_code == 200:
                        with open(save_path, 'wb') as f:
                            for chunk in img_resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
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
        logger.info(f"To run reconstruction: bash scripts/run_pipeline.sh {dataset_name}")
    else:
        logger.info("No new images found or downloaded.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_id = sys.argv[1]
        download_scan(target_id)
    else:
        print("Please provide a Scan ID.")
        print("Example: python download_robot_images.py 123e4567-e89b-...")