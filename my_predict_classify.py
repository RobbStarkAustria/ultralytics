from ultralytics import YOLO

model = YOLO("runs/classify/train4/weights/best.pt")
result = model(
    "test_images/clef_early_music_online_d177_00230.jpg",
    save=True
    
)
warte = ""