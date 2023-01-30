from ultralytics import YOLO

model = YOLO("diamant_notes/correct-boxes/weights/best.pt")
model.val(
    data="./data/test_images.yaml",
    device=0,
    imgsz=1024,
    batch=8,
    # conf=0.7,
    save_json=True,
    plots=True,
    v5loader=True
)
