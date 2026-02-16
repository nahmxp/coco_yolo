from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="coco_dataset/coco_dataset",
    save_dir="coco_dataset/yolo_dataset",
    use_keypoints=False,
)