from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load a custom model

# Evaluate on validation or test dataset
metrics = model.val(
    data="data.yaml",
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.7,
    device="0"
)