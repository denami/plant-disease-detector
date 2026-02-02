from ultralytics import YOLO
from pathlib import Path
from tabulate import tabulate

MODEL_PATH = "runs/classify/train/weights/best.pt"
TEST_DIR = Path("test")

model = YOLO(MODEL_PATH)
results = model(TEST_DIR)

table = []

for r in results:
    image_name = Path(r.path).name
    class_id = r.probs.top1
    confidence = r.probs.top1conf.item()
    class_name = r.names[class_id]

    status = "Diseased" if "healthy" not in class_name.lower() else "Healthy"

    table.append([
        image_name,
        class_name,
        status,
        f"{confidence:.2f}"
    ])

headers = ["Image", "Predicted class", "Status", "Confidence"]

print(tabulate(table, headers=headers, tablefmt="github"))
