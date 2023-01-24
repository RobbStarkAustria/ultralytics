from ultralytics import YOLO

model = YOLO("runs/detect/train8/weights/best.pt")
model.val(
    data="data/mi_d_mi_u_seils.yaml",
    device=0,
    imgsz=1024,
    batch=4,
    iou=0.3,
    # save_hybrid=True,
    # save_conf=True
)
