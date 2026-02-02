from ultralytics import YOLO
from pathlib import Path

def main():
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_DIR = BASE_DIR / "dataset"

    model = YOLO("yolo26s-cls.pt")

    model.train(
        data=DATASET_DIR,
        imgsz=224,
        epochs=30,
        batch=32,
        device=0,   # 0 для обучения используя GPU либо device="cpu",
        workers=0   # ← важно для Windows
    )

if __name__ == "__main__":
    main()
