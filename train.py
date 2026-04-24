"""
Train a YOLOv11 model on the US Football Ball Detection dataset.

Dataset: Roboflow COCO export at train_set/
Steps:
  1. Convert COCO annotations -> YOLO txt format
  2. Split images 80/20 into train/val
  3. Write data.yaml
  4. Train with Ultralytics YOLO

Usage:
    pip install ultralytics
    python train.py
"""

import json
import os
import random
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_ROOT = Path(__file__).parent / "train_set" / "train"
COCO_JSON    = DATASET_ROOT / "_annotations.coco.json"
OUTPUT_DIR   = Path(__file__).parent / "yolo_dataset"

MODEL        = "yolo11n.pt"   # pretrained checkpoint to fine-tune
EPOCHS       = 100
IMG_SIZE     = 640
BATCH        = 16
VAL_SPLIT    = 0.2
SEED         = 42

# ── COCO → YOLO conversion ────────────────────────────────────────────────────

def coco_to_yolo(coco_path: Path, output_dir: Path) -> tuple[list[Path], dict]:
    """Convert a COCO JSON to YOLO txt label files.

    Returns:
        image_paths: list of image Path objects (all images in the dataset)
        class_names: dict mapping YOLO class index -> name
    """
    with open(coco_path) as f:
        data = json.load(f)

    # category_id 0 is a background placeholder in Roboflow exports; real labels use id >= 1
    real_cats = [c for c in data["categories"] if c["id"] != 0]
    cat_map = {c["id"]: idx for idx, c in enumerate(real_cats)}
    class_names = {idx: c["name"] for idx, c in enumerate(real_cats)}

    id_to_image = {img["id"]: img for img in data["images"]}

    img_annotations: dict[int, list] = {img["id"]: [] for img in data["images"]}
    for ann in data["annotations"]:
        if ann["category_id"] in cat_map:
            img_annotations[ann["image_id"]].append(ann)

    labels_dir = output_dir / "labels_raw"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for img_meta in data["images"]:
        img_id   = img_meta["id"]
        img_file = DATASET_ROOT / img_meta["file_name"]
        if not img_file.exists():
            print(f"  [warn] image not found, skipping: {img_file.name}")
            continue
        image_paths.append(img_file)

        W, H = img_meta["width"], img_meta["height"]
        lines = []
        for ann in img_annotations[img_id]:
            yolo_cls = cat_map[ann["category_id"]]
            # bbox values may be stored as strings in Roboflow COCO exports
            x, y, w, h = [float(v) for v in ann["bbox"]]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_file = labels_dir / f"{img_file.stem}.txt"
        label_file.write_text("\n".join(lines))

    print(f"Converted {len(image_paths)} images, {len(data['annotations'])} annotations.")
    print(f"Classes: {class_names}")
    return image_paths, class_names


def split_and_copy(image_paths: list[Path], output_dir: Path):
    """Split images/labels 80/20 and copy into output_dir/{images,labels}/{train,val}."""
    random.seed(SEED)
    shuffled = image_paths.copy()
    random.shuffle(shuffled)

    n_val   = max(1, int(len(shuffled) * VAL_SPLIT))
    val_set = {p.name for p in shuffled[:n_val]}

    labels_raw = output_dir / "labels_raw"

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img_path in shuffled:
        split = "val" if img_path.name in val_set else "train"
        shutil.copy2(img_path, output_dir / "images" / split / img_path.name)

        label_src = labels_raw / f"{img_path.stem}.txt"
        label_dst = output_dir / "labels" / split / f"{img_path.stem}.txt"
        if label_src.exists():
            shutil.copy2(label_src, label_dst)
        else:
            label_dst.touch()

    train_count = len(shuffled) - n_val
    print(f"Split: {train_count} train / {n_val} val")


def write_data_yaml(output_dir: Path, class_names: dict) -> Path:
    names_list = [class_names[i] for i in range(len(class_names))]
    yaml_content = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(names_list)}\n"
        f"names: {names_list}\n"
    )
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Wrote {yaml_path}")
    return yaml_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not found. Install it with:\n"
            "    pip install ultralytics"
        )

    print("=== Step 1: Converting COCO → YOLO ===")
    image_paths, class_names = coco_to_yolo(COCO_JSON, OUTPUT_DIR)

    print("\n=== Step 2: Splitting dataset ===")
    split_and_copy(image_paths, OUTPUT_DIR)

    print("\n=== Step 3: Writing data.yaml ===")
    yaml_path = write_data_yaml(OUTPUT_DIR, class_names)

    print(f"\n=== Step 4: Training {MODEL} for {EPOCHS} epochs ===")
    model = YOLO(MODEL)
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(Path(__file__).parent / "runs"),
        name="football_ball",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
    )

    best_weights = Path(__file__).parent / "runs" / "football_ball" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")


if __name__ == "__main__":
    main()
