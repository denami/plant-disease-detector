from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from pathlib import Path
import shutil
import uuid

# --- настройки ---
MODEL_PATH = "runs/classify/train/weights/best.pt"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- инициализация ---
app = FastAPI(title="Plant Disease Detection")
model = YOLO(MODEL_PATH)

# --- HTML шаблон ---
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Plant Disease Detector</title>
    <style>
        body {{ font-family: Arial; margin: 40px; }}
        .box {{ max-width: 500px; margin: auto; }}
        .result {{ margin-top: 20px; padding: 10px; border-radius: 5px; }}
        .healthy {{ background-color: #d4edda; }}
        .diseased {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
<div class="box">
    <h2>🌿 Определение заболевания растения</h2>

    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br><br>
        <button type="submit">Загрузить и проверить</button>
    </form>

    {result}
</div>
</body>
</html>
"""

# --- главная страница ---
@app.get("/", response_class=HTMLResponse)
def main():
    return HTML_PAGE.format(result="")

# --- обработка изображения ---
@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    temp_name = f"{uuid.uuid4()}{suffix}"
    temp_path = UPLOAD_DIR / temp_name

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_path)
    r = results[0]

    class_id = r.probs.top1
    confidence = r.probs.top1conf.item()
    class_name = r.names[class_id]

    is_diseased = "healthy" not in class_name.lower()
    status = "Заражение обнаружено" if is_diseased else "Растение здорово"
    css_class = "diseased" if is_diseased else "healthy"

    temp_path.unlink(missing_ok=True)

    result_html = f"""
    <div class="result {css_class}">
        <strong>Результат:</strong> {status}<br>
        <strong>Класс:</strong> {class_name}<br>
        <strong>Уверенность:</strong> {confidence:.2f}
    </div>
    """

    return HTML_PAGE.format(result=result_html)
