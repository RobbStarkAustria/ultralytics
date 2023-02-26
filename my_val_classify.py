from ultralytics import YOLO

model = YOLO("clef/baseline/weights/best.pt")
metrics = model.val(
    data="../Seils_imslp/",
    epochs=100,
    imgsz=224,
    batch=100,
    plots=True,
    save_txt=True,
    save_conf=True,
)
warte = ""
print(metrics)