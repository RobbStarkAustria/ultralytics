from ultralytics import YOLO
from PIL import Image
import cv2

model1 = YOLO("diamant_notes/correct-boxes/weights/best.pt")

# im1 = Image.open("test_images/alberti_dalmio_A.jpg")
im1 = cv2.imread("test_images/bertani_4.jpg")
# im2 = cv2.imread("test_images/correggio_mentreil_A.jpg")

# results1 = model1.predict(source=[im1, im2], save=True, line_thickness=1)

model2 = YOLO("diamant_notes/correct-boxes/weights/best.pt")

results2 = model2.predict(source=[im1],
                          save=True,
                          line_thickness=1,
                          imgsz=1024,
                          #   conf=0.25,
                          #   iou=0,
                          save_txt=True,
                          save_conf=True,
                          )
warte = ""
