import os
import json
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image
from tqdm import tqdm

# ImageNet normalization parameters
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def process_single_image(idx, img_entry, output_dir, resolution, zod_root):
    """Worker task to process a single image frame using a 0-based index."""
    img_id = img_entry["id"]
    img_path = os.path.join(zod_root, img_entry["file_name"])

    if not os.path.exists(img_path):
        return None, f"Warning: Could not find image at {img_path}, skipping."

    try:
        # 1. Open with PIL to match RF-DETR native decoder
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # PIL size is (width, height)

        # 2. Resize using PIL Bilinear matching PyTorch/PIL backend
        img_resized = img.resize((resolution, resolution), Image.Resampling.BILINEAR)

        # 3. Convert to float numpy array and scale to [0, 1]
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        # 4. ImageNet Normalization
        img_tensor = (img_np - MEAN) / STD

        # 5. Transpose HWC -> CHW
        img_tensor = np.transpose(img_tensor, (2, 0, 1))

        # 6. Save binary array as a pickle file: val_{idx}.pkl
        out_name = f"val_{idx}.pkl"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "wb") as f:
            pickle.dump(img_tensor, f)

        # Return metadata tuple keyed by index, referencing original COCO ID
        meta_item = (
            str(idx),
            {
                "file_name": img_path,
                "orig_shape": [orig_h, orig_w],
                "coco_image_id": img_id,
                "sample_idx": idx,
            },
        )
        return meta_item, None

    except Exception as e:
        return None, f"Warning: Failed to process image at {img_path}, skipping. Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Preprocess COCO validation dataset for MLPerf RF-DETR")
    parser.add_argument(
        "--zod_root",
        type=str,
        required=True,
        help="Path to the ZOD dataset root directory"
    )
    parser.add_argument(
        "--annotation_file", 
        type=str, 
        required=True, 
        help="Path to COCO validation JSON containing absolute image paths"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Where to save .pkl files"
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=640, 
        help="Target model resolution"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=os.cpu_count() or 4, 
        help="Number of CPU worker processes (default: all available logical CPUs)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load COCO ground truth JSON
    with open(args.annotation_file, "r") as f:
        coco_gt = json.load(f)

    images = coco_gt.get("images", [])
    print(f"Loaded {len(images)} images from {args.annotation_file}")
    print(f"Parallelizing preprocessing using {args.workers} CPU workers...")

    metadata = {}
    success_count = 0

    # Parallelize processing across CPU cores with zero-based index mapping
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_single_image, idx, img_entry, args.output_dir, args.resolution, args.zod_root)
            for idx, img_entry in enumerate(images)
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing Images", unit="img"):
            meta_item, warning_msg = future.result()
            
            if warning_msg:
                tqdm.write(warning_msg)
                
            if meta_item:
                key, val = meta_item
                metadata[key] = val
                success_count += 1

    # Save aggregated metadata mapping
    with open(os.path.join(args.output_dir, "mlperf_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSuccessfully preprocessed {success_count}/{len(images)} images -> Saved to {args.output_dir}")


if __name__ == "__main__":
    main()