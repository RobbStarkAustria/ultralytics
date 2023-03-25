from ultralytics import YOLO

model = YOLO("diamant_notes/2023_03_17_no_aug_no_patience/weights/best.pt")
model.val(
    data="data/diamant_notes.yaml",
    device=0,
    imgsz=1024,
    batch=10,
    # conf=0.7,
    save_json=False,
    plots=True,
    # v5loader=True
)
