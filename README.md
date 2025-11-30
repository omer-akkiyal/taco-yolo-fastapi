# TACO YOLOv8 + FastAPI + Web UI (Litter Detection Demo)

Bu repo, **TACO (Trash Annotations in Context)** veri seti ile YOLOv8 nesne tespiti eğitimi (fine-tune) yapıp, çıkan modeli **FastAPI** üzerinden `/predict` endpoint’i ile servis eder. Ayrıca aynı endpoint’i kullanan basit bir **Web UI** içerir.

> Not: Dataset ve `runs/` gibi eğitim çıktıları repo’ya eklenmez. Model ağırlığı (weights) **GitHub Releases** üzerinden paylaşılır.

---

## Özellikler
- TACO `annotations.json` → YOLO formatına dönüşüm (`convert_taco_to_yolo.py`)
- YOLOv8 ile hızlı fine-tune (demo seviyesinde)
- FastAPI servis:
  - `GET /health`
  - `POST /predict` (image upload → JSON bbox/class/confidence)
- UI:
  - Görsel yükle
  - `conf` slider ile threshold ayarla
  - BBox çizimi (canvas) + JSON görüntüleme

---

## Proje Yapısı (özet)
```
.
├─ frontend/
│  ├─ index.html
│  ├─ styles.css
│  └─ app.js
├─ main.py
├─ convert_taco_to_yolo.py
├─ requirements.txt
└─ README.md
```

Repo’ya dahil edilmeyenler: `data/`, `yolo_dataset/`, `runs/`, `.venv/`, büyük model dosyaları.

---

## Kurulum
### 1) Sanal ortam (önerilir)
Windows PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Paketler
```powershell
py -m pip install -r requirements.txt
```

---

## Model Ağırlıkları (Weights)
Bu repo ağırlıkları doğrudan commit etmez.

1) GitHub repo sayfasında **Releases** bölümünden modeli indirin (`best.pt` önerilir)
2) Repo kökünde `models/` klasörü oluşturup içine koyun:

```
models/best.pt
```

3) API’yi şu şekilde başlatın:

```powershell
$env:MODEL_PATH="models\best.pt"
py -m uvicorn main:app
```

> Eğer `MODEL_PATH` verilmezse `main.py` içindeki otomatik tarama/varsayılan yol kullanılır.

---

## TACO Dataset → YOLO Format (opsiyonel)
Bu adım, kendi ortamınızda yeniden üretmek içindir.

- Kaggle TACO dataset içeriğini proje köküne `data/` klasörü olarak koyun:
  - `data/annotations.json`
  - `data/batch_*/...jpg`

Sonra:
```powershell
py .\convert_taco_to_yolo.py
```

Çıktı:
- `yolo_dataset/` (images/labels + `data.yaml`)

---

## Eğitim (Train) (opsiyonel)
Demo için kısa eğitim:
```powershell
yolo detect train data=.\yolo_dataset\data.yaml model=yolov8n.pt epochs=5 imgsz=640 batch=8
```

Eğitim çıktısı:
- `runs/detect/train/weights/best.pt`

---

## API’yi Çalıştırma
```powershell
py -m uvicorn main:app
```

- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

### Predict (Swagger ile)
`/docs` → `/predict` → **Try it out** → image upload → Execute

### Predict (curl)
```powershell
curl -X POST "http://127.0.0.1:8000/predict?conf=0.25" -F "file=@.\some_image.jpg"
```

---

## Web UI
UI FastAPI tarafından `/ui` altında servis edilir:

- UI: `http://127.0.0.1:8000/ui`

Kullanım:
1) Dosya seçin
2) `conf` ayarlayın
3) **Tahmin Et** → bbox + class + confidence görünecek

---

## Çıktı Formatı (örnek)
`/predict` JSON yanıtı:
```json
{
  "count": 2,
  "detections": [
    {
      "class_id": 12,
      "class_name": "Drink can",
      "confidence": 0.63,
      "box_xyxy": [120.5, 210.2, 260.8, 355.9]
    }
  ]
}
```
---

## Lisans / Kaynak
- Dataset: TACO (Trash Annotations in Context)
- Eğitim/Inference kütüphanesi: Ultralytics YOLOv8
