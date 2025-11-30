from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile, os
from pathlib import Path
import requests

app = FastAPI(title="TACO YOLO API")

# CORS (demo için açık; isterseniz Netlify domaininizi yazın)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render’da da UI isterseniz kalsın (Netlify kullanacaksanız şart değil)
if Path("frontend").exists():
    app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")

# ---- MODEL WEIGHTS ----
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
MODEL_URL = os.getenv("MODEL_URL", "https://github.com/omer-akkiyal/taco-yolo-fastapi/releases/download/v0.1/best.pt")


def ensure_weights():
    p = Path(MODEL_PATH)
    if p.exists():
        return

    if not MODEL_URL:
        raise RuntimeError(
            f"Model weights not found at {MODEL_PATH}. "
            f"Set MODEL_PATH to an existing file or set MODEL_URL to download it."
        )

    # indir
    r = requests.get(MODEL_URL, stream=True, timeout=180)
    r.raise_for_status()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

# weights’i garanti et, sonra modeli yükle
ensure_weights()
model = YOLO(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH, "weights_exists": Path(MODEL_PATH).exists()}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen bir image dosyası yükleyin.")

    suffix = "." + (file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        results = model.predict(tmp_path, conf=conf, verbose=False)
        r = results[0]

        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box, cls_id, c in zip(
                r.boxes.xyxy.tolist(),
                r.boxes.cls.tolist(),
                r.boxes.conf.tolist()
            ):
                cls_int = int(cls_id)
                detections.append({
                    "class_id": cls_int,
                    "class_name": model.names.get(cls_int, str(cls_int)),
                    "confidence": float(c),
                    "box_xyxy": [float(x) for x in box]
                })

        return {"count": len(detections), "detections": detections}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
