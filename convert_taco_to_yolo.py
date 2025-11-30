import json
import random
import shutil
from pathlib import Path
from PIL import Image

# ---------- Ayarlar ----------
ROOT = Path(__file__).resolve().parent
TACO_DATA = ROOT / "data"
ANN_PATH = TACO_DATA / "annotations.json"

OUT = ROOT / "yolo_dataset"
TRAIN_RATIO = 0.8
SEED = 42

# Çok sınıf varsa ve yetişmek istiyorsanız bunu 10 gibi sınırlandırabilirsiniz.
LIMIT_TOP_K_CLASSES = None  # örn: 10  veya None

# ---------- Yardımcılar ----------
def find_image_path(file_name: str) -> Path:
    # annotations.json içindeki file_name bazen "batch_1/xxx.jpg" gibi gelir
    candidate = TACO_DATA / file_name
    if candidate.exists():
        return candidate

    # bazen sadece "xxx.jpg" gelir; batch'lerde ara
    base = Path(file_name).name
    hits = list(TACO_DATA.glob(f"batch_*/{base}"))
    if hits:
        return hits[0]

    raise FileNotFoundError(f"Görüntü bulunamadı: {file_name}")

def coco_bbox_to_yolo(bbox, img_w, img_h):
    # bbox: [x_min, y_min, width, height] (piksel)
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return None
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # clamp
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    # çok küçük kalanları ele
    if w_norm <= 0 or h_norm <= 0:
        return None
    return x_center, y_center, w_norm, h_norm

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- Ana ----------
random.seed(SEED)

with open(ANN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}
annotations = data["annotations"]
categories = {c["id"]: c["name"] for c in data["categories"]}

# Sınıf id'lerini YOLO için 0..N-1'e remap edelim
# İsterseniz en çok görünen K sınıfı seçelim
from collections import Counter, defaultdict
cat_counts = Counter([a["category_id"] for a in annotations])

selected_cats = list(categories.keys())
if LIMIT_TOP_K_CLASSES is not None:
    selected_cats = [cid for cid, _ in cat_counts.most_common(LIMIT_TOP_K_CLASSES)]

# remap
selected_cats = sorted(selected_cats, key=lambda cid: categories.get(cid, str(cid)))
cat_remap = {cid: i for i, cid in enumerate(selected_cats)}
names = [categories[cid] for cid in selected_cats]

ann_by_image = defaultdict(list)
for a in annotations:
    cid = a["category_id"]
    if cid in cat_remap:
        ann_by_image[a["image_id"]].append(a)

# output klasörleri
img_all = OUT / "images_all"
lbl_all = OUT / "labels_all"
safe_mkdir(img_all)
safe_mkdir(lbl_all)

kept = 0
skipped = 0

for image_id, img in images.items():
    file_name = img.get("file_name")
    if not file_name:
        continue

    # Bu görüntünün annotation'ı yoksa da dahil edelim mi?
    anns = ann_by_image.get(image_id, [])
    # YOLO eğitiminde boş label dosyası olabilir; sorun değil.
    try:
        src_img_path = find_image_path(file_name)
    except FileNotFoundError:
        skipped += 1
        continue

    # width/height yoksa görselden okuyalım
    img_w = img.get("width")
    img_h = img.get("height")
    if not img_w or not img_h:
        with Image.open(src_img_path) as im:
            img_w, img_h = im.size

    # hedef adı (batch klasörleri çakışmasın diye image_id kullanalım)
    out_stem = str(image_id).zfill(8)
    out_img_path = img_all / f"{out_stem}{src_img_path.suffix.lower()}"
    out_lbl_path = lbl_all / f"{out_stem}.txt"

    # image kopyala
    shutil.copy2(src_img_path, out_img_path)

    # label yaz
    lines = []
    for a in anns:
        bbox = a.get("bbox")
        if not bbox:
            continue
        yolo_box = coco_bbox_to_yolo(bbox, img_w, img_h)
        if yolo_box is None:
            continue
        cls_id = cat_remap[a["category_id"]]
        x_center, y_center, w_norm, h_norm = yolo_box
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    out_lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    kept += 1

print(f"✅ Dönüşen image: {kept}, atlanan: {skipped}")
print(f"✅ Sınıf sayısı: {len(names)}")

# train/val split
all_imgs = sorted(img_all.glob("*.*"))
random.shuffle(all_imgs)
split = int(len(all_imgs) * TRAIN_RATIO)
train_imgs = all_imgs[:split]
val_imgs = all_imgs[split:]

# final klasör yapısı
for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
    safe_mkdir(OUT / "images" / split_name)
    safe_mkdir(OUT / "labels" / split_name)
    for p in split_imgs:
        target_img = OUT / "images" / split_name / p.name
        shutil.move(str(p), str(target_img))

        label_src = lbl_all / (p.stem + ".txt")
        target_lbl = OUT / "labels" / split_name / (p.stem + ".txt")
        shutil.move(str(label_src), str(target_lbl))

# temizlik
try:
    img_all.rmdir()
    lbl_all.rmdir()
except OSError:
    pass

# data.yaml yaz
yaml = OUT / "data.yaml"
yaml.write_text(
f"""path: {OUT.as_posix()}
train: images/train
val: images/val

names:
""" + "\n".join([f"  {i}: {n}" for i, n in enumerate(names)]) + "\n",
encoding="utf-8"
)

print(f"✅ Hazır: {yaml}")
print("📌 Sonraki adım: yolo detect train data=yolo_dataset/data.yaml model=yolov8n.pt epochs=5 imgsz=640")
