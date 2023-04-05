import pathlib
import os
import torch
import pandas as pd
from ultralytics import YOLO
from more_itertools import chunked
from tabulate import tabulate


def calculate_validation(model_path):
    model = YOLO(model_path)
    pitches = model.names

    home_folder = os.path.expanduser("~")

    val_path = os.path.join(home_folder, "PycharmProjects", "datasets", "classify", "val")
    val_dir = pathlib.Path(val_path)
    val_dir_list = []
    for dir in val_dir.iterdir():
        val_dir_list.append(os.path.basename(dir))

    evaluations = {}
    whole_correct = 0
    whole_results = 0
    for pitch in val_dir_list:
        pitch_path = os.path.join(val_path, pitch)
        results = model(
            source=pitch_path,
            imgsz=352,
            batch=100
        )
        correct_pitch = 0
        for r in results:
            value = torch.argmax(r.probs)
            pred_pitch = pitches[value.item()]

            if pred_pitch == pitch:
                correct_pitch += 1

    # print(f"{correct_pitch/len(results)}")
        evaluations[pitch] = {
            "count": len(results),
            "correct": correct_pitch,
            "ratio": correct_pitch / len(results)
        }

        whole_correct += correct_pitch
        whole_results += len(results)

    evaluations["whole"] = {
        "count": whole_results,
        "correct": whole_correct,
        "ratio": whole_correct / whole_results
    }

    experiment_path = str.split(model_path, "/weights")[0]
    csv_path = os.path.join(experiment_path, "val_results.csv")
    df = pd.DataFrame(evaluations)
    df2 = df.T
    df2.to_csv(csv_path, header=True)
    print(f"model: {model_path}")
    headers = ["pitch", "count", "correct", "ratio"]
    print(tabulate(df.T, headers="keys"))
    warte = ""


if __name__ == "__main__":
    model_path = "minima_down/all_notes_346_0.00001/weights/best.pt"
    calculate_validation(model_path)
