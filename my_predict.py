from ultralytics import YOLO

model = YOLO("diamant_notes/v5_max_epochs_m_model/weights/best.pt")

results = model.predict(
    source="test_images/bertani_4.jpg",
    device=0,
    imgsz=1024,
    batch=4,
    conf=0.7,
    iou=0.1,
    max_det=1000,
    # multi_label=False,
    save=True,
    save_txt=True,
    save_conf=True,
    line_thickness=1,
    # agnostic_nms=True
)