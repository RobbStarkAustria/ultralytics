from ultralytics import YOLO

model = YOLO("extra_symbols/max_epochs_m_model/weights/best.pt")
model.val(
    data="./data/extra_symbols_val_seils.yaml",
    device=0,
    imgsz=1024,
    batch=8,
    # conf=0.7,
    save_json=True,
    plots=True,
    # v5loader=True
)
