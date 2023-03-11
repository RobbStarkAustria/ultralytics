import os
from ultralytics import YOLO

with open("watchdog.txt", "w") as f:
    f.write("running")
    print("watchdog ready!")

model = YOLO("mensuration/baseline/weights/best.pt")
project = "mensuration"
name = "all_notes"
model.train(
    project=project,
    name=name,
    data="../datasets/classify",
    epochs=150,
    imgsz=224,
    batch=100,
    patience=20,
    optimizer="Adam",
    lr0=0.001,
    cos_lr=True,
    plots=True,
    cls=0.5,
    dfl=1.5,
    fl_gamma=0.0,
    label_smoothing=0.0,
    nbs=64,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0
)

# model.val(
#     project=project,
#     name=name,
#     data="../datasets/classify",
#     epochs=100,
#     imgsz=224,
#     batch=100,
#     plots=True,
#     save_txt=True,
#     save_conf=True,
#     fl_gamma=0.0,
#     label_smoothing=0.0,
#     nbs=64,
#     hsv_h=0.0,
#     hsv_s=0.0,
#     hsv_v=0.0,
#     degrees=0.0,
#     translate=0.0,
#     scale=0.0,
#     shear=0,
#     perspective=0.0,
#     flipud=0.0,
#     fliplr=0.0,
#     mosaic=0.0,
#     mixup=0.0,
#     copy_paste=0.0
# )

os.unlink("watchdog.txt")
print("watchdog sleeping")