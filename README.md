# COCO to YOLO Dataset Converter

This project provides a script to convert object detection datasets from the [COCO format](https://cocodataset.org/#format-data) to the [YOLO format](https://docs.ultralytics.com/datasets/detect/#format). It also splits the dataset into train, validation, and test sets, and generates a `dataset.yaml` file for easy use with YOLO training frameworks.

## Features

- **Automatic extraction**: Unzips a COCO dataset archive.
- **Flexible splitting**: Randomly splits images into train/val/test sets (default: 70/20/10).
- **Annotation conversion**: Converts COCO bounding boxes and polygons to YOLO format.
- **Image copying**: Copies images into the correct YOLO directory structure.
- **YOLO config**: Generates a `dataset.yaml` file for training.

## Project Structure

```
.
├── cnv.py                # Main conversion script
├── coco_dataset.zip      # Your zipped COCO dataset (input)
├── yolo_dataset/         # Output folder (created by script)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── dataset.yaml
```

## Requirements

- Python 3.7+
- [Pillow](https://pypi.org/project/Pillow/)
- [tqdm](https://pypi.org/project/tqdm/)
- [pyyaml](https://pypi.org/project/PyYAML/)
- numpy

Install dependencies with:
```bash
pip install pillow tqdm pyyaml numpy
```

## Usage

1. **Place your COCO dataset ZIP** (must contain `coco.json` and images) in the project folder as `coco_dataset.zip`.

2. **Run the script:**
   ```bash
   python cnv.py
   ```

3. **Result:**  
   The converted YOLO dataset will be in the `yolo_dataset/` folder, ready for training.

## Notes

- The script will automatically extract `coco_dataset.zip`, find `coco.json`, and process all images and annotations.
- If image width/height is missing in the JSON, it will read the image file to get the size.
- Both bounding boxes and segmentation polygons are supported (polygons are converted to normalized YOLO format).

## License

MIT License

---

**Author:**  
Your Name  
