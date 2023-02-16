from ultralytics import YOLO

model = YOLO("runs/classify/train4/weights/best.pt")
model.val(
    data="../datasets/classify",
    epochs=100,
    imgsz=224,
    batch=100,
    plots=True,
    save_txt=True,
    save_conf=True,
)