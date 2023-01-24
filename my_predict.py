from ultralytics import YOLO

model = YOLO("runs/detect/train8/weights/best.pt")
results = model.predict(
    source="../datasets/note_sheets/images/test",
    device=0,
    imgsz=1024,
    batch=4,
    iou=0.3,
    save=True,
    save_txt=True,
    save_conf=True,
    line_thickness=2,
    # pbar=True
)