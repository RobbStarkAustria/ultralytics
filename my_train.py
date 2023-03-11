import os
from ultralytics import YOLO

import my_config

classes_string = "_".join(my_config.classes_to_train).replace("-", "_")

if len(my_config.note_group) >0:
        classes_string = my_config.note_group

with open("watchdog.txt", "w") as f:
    f.write("running")
    print("watchdog ready!")

name = "2023_02_28_with_microfiches_transfer_aug_1_new_fitness"

# model = YOLO("yolov8m.pt")
model = YOLO("extra_symbols/baseline_m_min_memory_aug_1/weights/best.pt")
model.train(
    data=f"./data/{classes_string}.yaml",
    project=f"{classes_string}",
    # project="hyper-parameter",
    name=f"{name}",
    batch=-1,
    min_memory=True,
    patience=20,
    epochs=150,
    device=0,
    imgsz=1024,
    optimizer="Adam",
    max_det=1000,
    lr0=0.001,
    cos_lr=True,
    augment=False,
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
    name=f"{name}-val",
    save_txt=True,
    save_conf=True,
    device=0,
    imgsz=1024,
    batch=4,
)

os.unlink("watchdog.txt")
print("watchdog sleeping")
