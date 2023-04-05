from ultralytics import YOLO

model = YOLO("minima_down/all_notes_346/weights/best.pt")
metrics = model.val(
    data="../datasets/classify",
    # epochs=100,
    imgsz=352,
    batch=100,
    plots=True,
    save_txt=True,
    save_conf=True,
)
warte = ""
print(metrics)