import os
from ultralytics import YOLO

import my_config

classes_string = "_".join(my_config.classes_to_train).replace("-", "_")

with open("watchdog.txt", "w") as f:
    f.write("running")
    print("watchdog ready!")

model = YOLO("yolov8x.pt")
model.train(
    data=f"./data/{classes_string}.yaml",
    project=f"{classes_string}",
    # project="hyper-parameter",
    name="initial",
    batch=2,
    patience=10,
    epochs=150,
    device=0,
    imgsz=1024,
    optimizer="Adam",
    max_det=1000,
    lr0=0.001,
    cos_lr=True,
    augment=True,
    plots=True,
    cls=0.5,
    dfl=1.5,
    fl_gamma=0.0,
    label_smoothing=0.0,
    nbs=64,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.0,
    shear=0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    )

model.val(
    save_txt=True,
    save_conf=True,
    device=0,
    imgsz=1024,
    batch=4,
    iou=0.3,
)

os.unlink("watchdog.txt")
print("watchdog sleeping")

