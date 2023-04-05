import os
import sys

import torch
from ultralytics import YOLO
from my_validation_classify import calculate_validation

home_path = os.path.expanduser('~')

sys.path.append(home_path + "/PycharmProjects/transfer_pytorch")

from cuda_helpers import free_memory

with open("watchdog.txt", "w") as f:
    f.write("running")
    print("watchdog ready!")

model = YOLO("minima_up/baseline/weights/best.pt")
project = "minima_up"
name = "all_notes_352_0.0001"
model.train(
    project=project,
    name=name,
    data="../datasets/classify",
    cache="ram",
    epochs=150,
    imgsz=352,
    batch=-1,
    patience=20,
    optimizer="Adam",
    lr0=0.0001,
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

if torch.cuda.is_available():
        free_memory(to_delete=["0"])
        
model_path = os.path.join(project, name, "weights", "best.pt")
calculate_validation(model_path)

os.unlink("watchdog.txt")
print("watchdog sleeping")