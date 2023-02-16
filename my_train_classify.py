from ultralytics import YOLO

model = YOLO("yolov8m-cls.pt")
model.train(
    data="../datasets/classify",
    epochs=100,
    imgsz=224,
    batch=100,
)