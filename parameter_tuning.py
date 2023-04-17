import os
# import init
from ray import tune
from ultralytics import YOLO
from timeit import default_timer as timer


def main():
    # experiment_name = "lr-, dfl-tuning"
    # transfer_model = "yolov8m.pt"
    # basemodel = YOLO(transfer_model)

    with open("watchdog.txt", "w") as f:
        f.write("running")
        print("watchdog ready!")

    model = YOLO('yolov8n.pt')
    result = model.tune(data="diamant_notes.yaml", space={"epochs": tune.choice([2, 3, 6]),
                                                    "lr0": tune.uniform(0.001, 0.1),
                                                    "imgsz": 1024}, max_samples=3)

    # result = basemodel.tune(
    #     data="data/diamant_notes.yaml",
    #     space={
    #         # "epochs": tune.choice([25, 50, 65]),
    #         "lr0": tune.uniform(0.00001, 0.001),
    #         # "dfl": tune.uniform(1, 5)
    #     },
    #     max_samples=2,
    #     # device=0
    #     # project="tuning",
    #     # name=f"{experiment_name}",
    #     # exist_ok=True,
    #     # cache="ram",
    #     # batch=5,
    #     # patience=0,
    #     # # epochs=65,
    #     # device=0,
    #     # imgsz=1024,
    #     # optimizer="AdamW",
    #     # max_det=1000,
    #     # # lr0=0.0001,
    #     # cos_lr=True,
    #     # augment=False,
    #     # plots=True,
    #     # cls=0.5,
    #     # # dfl=1.5,
    #     # label_smoothing=0.0,
    #     # nbs=64,
    #     # hsv_h=0.015,
    #     # hsv_s=0.7,
    #     # hsv_v=0.4,
    #     # degrees=0.0,
    #     # translate=0,
    #     # scale=0.0,
    #     # shear=0,
    #     # perspective=0.0,
    #     # flipud=0.0,
    #     # fliplr=0.0,
    #     # mosaic=0.5,
    #     # mixup=0.0,
    #     # copy_paste=0.0,
    # )

    os.unlink("watchdog.txt")
    print("watchdog sleeping")


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    print(f"Total Training time: {end - start}")
