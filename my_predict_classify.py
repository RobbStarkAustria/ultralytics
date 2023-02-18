from ultralytics import YOLO
import torch

model = YOLO("runs/classify/train7/weights/best.pt")
result = model(
    "test_images/sm-d",
    save=True
    
)
names = model.names
value = torch.argmax(result[0].probs)
print(names[value.item()])
warte = ""