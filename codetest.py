import ultralytics
from ultralytics import YOLO, settings

ultralytics.checks()
print(settings)

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data='coco8.yaml', epochs=3, auto_augment=None, batch=32)  # train the model

