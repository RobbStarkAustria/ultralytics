import ultralytics
from ultralytics import YOLO, settings

ultralytics.checks()
print(settings)

model = YOLO("yolo11l.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data='coco8.yaml', epochs=3, auto_augment=None, batch=0.8)  # train the model

