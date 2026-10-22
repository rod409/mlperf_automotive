import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate MLPerf LoadGen Accuracy for RF-DETR")
    parser.add_argument(
        "--mlperf-accuracy-file",
        required=True,
        help="Path to mlperf_log_accuracy.json")
    parser.add_argument(
        "--metadata-file",
        required=True,
        help="Path to mlperf_metadata.json generated during preprocessing")
    parser.add_argument(
        "--annotation-file",
        required=True,
        help="COCO format annotation file")
    return parser.parse_args()

def convert_to_coco_format(image_id, boxes_xyxy, scores, labels, category_mapping):
    """Converts xyxy boxes to COCO format: [x, y, width, height]"""
    results = []
    for box, score, label in zip(boxes_xyxy, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        coco_cat_id = category_mapping[int(label)]
        
        results.append({
            "image_id": int(image_id),
            "category_id": coco_cat_id,
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(score)
        })
    return results

def main():
    args = get_args()

    # 1. Initialize COCO Ground Truth and Category Mapping
    coco_gt = COCO(args.annotation_file)
    cat_ids = sorted(coco_gt.getCatIds())
    category_mapping = {i: cat_id for i, cat_id in enumerate(cat_ids)}
    num_valid_classes = len(category_mapping)

    # 2. Load Metadata Dictionary as a key-value lookup map
    with open(args.metadata_file, "r") as f:
        metadata = json.load(f)

    # 3. Load MLPerf Accuracy Log
    with open(args.mlperf_accuracy_file, "r") as f:
        accuracy_logs = json.load(f)

    seen = set()
    coco_format_results = []

    for j in tqdm(accuracy_logs, desc="Processing MLPerf Log"):
        idx = j['qsl_idx']
        
        # De-dupe in case LoadGen sends duplicate queries
        if idx in seen:
            continue
        seen.add(idx)

        # Direct string key lookup: qsl_idx -> "val_{qsl_idx}.npy" -> metadata[str(qsl_idx)]
        meta = metadata[str(idx)]
        img_id = meta["coco_image_id"]
        orig_h, orig_w = meta["orig_shape"]

        # Parse prediction bytes from LoadGen log
        data = np.frombuffer(bytes.fromhex(j['data']), dtype=np.float32)
        prediction = data.copy().reshape(-1, 6)  # [cx, cy, w, h, score, label]

        boxes_raw = prediction[:, :4]
        scores = prediction[:, 4]
        labels = prediction[:, 5]

        # 1. Filter out background class
        valid_mask = (labels < num_valid_classes) & (scores > 0.001)
        boxes = boxes_raw[valid_mask]
        scores = scores[valid_mask]
        labels = labels[valid_mask]

        if len(scores) == 0:
            continue

        # 2. Unscale cxcywh (0-1) to original image dimensions in xyxy format
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        x1 = (cx - 0.5 * w) * orig_w
        y1 = (cy - 0.5 * h) * orig_h
        x2 = (cx + 0.5 * w) * orig_w
        y2 = (cy + 0.5 * h) * orig_h

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)
        
        coco_format_results.extend(
            convert_to_coco_format(img_id, boxes_xyxy, scores, labels, category_mapping)
        )

    # 4. Run COCO Evaluation
    print("\n=== Evaluating COCO mAP ===")
    coco_dt = coco_gt.loadRes(coco_format_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    print(f"\nFinal MLPerf Accuracy mAP: {coco_eval.stats[0]:.8f}")

if __name__ == "__main__":
    main()