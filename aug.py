import os
import shutil
from pathlib import Path
import albumentations as A
import cv2
from tqdm import tqdm


def parse_yolo_label(line):
    parts = line.strip().split()
    cls = int(parts[0])
    coords = list(map(float, parts[1:]))
    if len(coords) == 4:  # bbox
        return {"class": cls, "bbox": coords, "poly": None}
    else:  # polygon
        poly = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        return {"class": cls, "bbox": None, "poly": poly}


def save_yolo_label(path, labels):
    with open(path, "w") as f:
        for lab in labels:
            if lab["bbox"] is not None:
                cls, (cx, cy, w, h) = lab["class"], lab["bbox"]
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            elif lab["poly"] is not None:
                cls = lab["class"]
                poly_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in lab["poly"]])
                f.write(f"{cls} {poly_str}\n")


def augment_dataset(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    aug_list = [
        ("hflip", A.HorizontalFlip(p=1.0)),
        ("vflip", A.VerticalFlip(p=1.0)),
        ("rot90", A.Rotate(limit=(90, 90), p=1.0)),
        ("rot180", A.Rotate(limit=(180, 180), p=1.0)),
        ("rot270", A.Rotate(limit=(270, 270), p=1.0)),
    ]

    for split in ["train", "val", "test"]:
        img_dir = Path(input_dir) / "images" / split
        lbl_dir = Path(input_dir) / "labels" / split
        out_img_dir = Path(output_dir) / "images" / split
        out_lbl_dir = Path(output_dir) / "labels" / split
        
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        print(f"Found {len(img_files)} images in {img_dir}")

        for img_file in tqdm(img_files, desc=f"Aug {split}"):
            label_file = lbl_dir / (img_file.stem + ".txt")
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Warning: Could not read image {img_file}")
                continue
            h, w = image.shape[:2]

            labels = []
            if label_file.exists():
                with open(label_file, "r") as f:
                    for line in f:
                        labels.append(parse_yolo_label(line))

            # Save original
            cv2.imwrite(str(out_img_dir / img_file.name), image)
            if label_file.exists():
                shutil.copy(label_file, out_lbl_dir / label_file.name)

            for name, aug in aug_list:
                aug_img_path = out_img_dir / f"{img_file.stem}_{name}.jpg"
                aug_lbl_path = out_lbl_dir / f"{img_file.stem}_{name}.txt"

                # Separate bboxes and polygons
                bboxes, classes_bbox = [], []
                keypoints, keypoints_cls, poly_splits = [], [], []

                for lab in labels:
                    if lab["bbox"] is not None:
                        cx, cy, bw, bh = lab["bbox"]
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        x2 = (cx + bw/2) * w
                        y2 = (cy + bh/2) * h
                        bboxes.append([x1, y1, x2, y2])
                        classes_bbox.append(lab["class"])
                    elif lab["poly"] is not None:
                        abs_poly = [(px * w, py * h) for px, py in lab["poly"]]
                        start_idx = len(keypoints)
                        for p in abs_poly:
                            keypoints.append(p)
                            keypoints_cls.append(lab["class"])
                        poly_splits.append((lab["class"], start_idx, len(abs_poly)))

                transform = A.Compose(
                    [aug],
                    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["classes_bbox"]),
                    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, label_fields=["keypoints_cls"])
                )

                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    classes_bbox=classes_bbox,
                    keypoints=keypoints,
                    keypoints_cls=keypoints_cls
                )

                aug_img = transformed["image"]
                aug_bboxes = transformed["bboxes"]
                aug_classes_bbox = transformed["classes_bbox"]
                aug_keypoints = transformed["keypoints"]

                cv2.imwrite(str(aug_img_path), aug_img)

                # Rebuild labels
                new_labels = []
                for bbox, cls in zip(aug_bboxes, aug_classes_bbox):
                    x1, y1, x2, y2 = bbox
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    new_labels.append({"class": cls, "bbox": [cx, cy, bw, bh], "poly": None})

                # Reconstruct polygons from keypoints
                idx = 0
                for cls, start_idx, length in poly_splits:
                    poly = aug_keypoints[start_idx:start_idx+length]
                    norm_poly = [(px / w, py / h) for px, py in poly]
                    new_labels.append({"class": cls, "bbox": None, "poly": norm_poly})
                    idx += length

                save_yolo_label(aug_lbl_path, new_labels)

    print(f"\n✅ Augmentation complete! Augmented dataset saved in {output_dir}")


if __name__ == "__main__":
    augment_dataset("yolo_dataset", "yolo_dataset_aug")
