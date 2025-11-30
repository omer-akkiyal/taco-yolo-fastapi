from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import tempfile, os
from pathlib import Path

app = FastAPI(title="TACO YOLO API")

app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")

MODEL_PATH = os.getenv("MODEL_PATH", r"runs\detect\train\weights\best.pt")
model = YOLO(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image dosyası yükleyin.")

    suffix = "." + (file.filename.split(".")[-1] if "." in file.filename else "jpg")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = model.predict(tmp_path, conf=conf, verbose=False)
        r = results[0]

        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box, cls_id, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                detections.append({
                    "class_id": int(cls_id),
                    "class_name": model.names.get(int(cls_id), str(int(cls_id))),
                    "confidence": float(c),
                    "box_xyxy": [float(x) for x in box]
                })

        return {"count": len(detections), "detections": detections}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
