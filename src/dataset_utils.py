"""Utilidades de dataset: descarga, extracción, configuración y visualización."""
import random
import urllib.request
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml

CLASS_NAMES = {0: "Green", 1: "off", 2: "red", 3: "wait_on", 4: "yellow"}
# Colores BGR por clase (OpenCV)
CLASS_COLORS = {
    0: (0, 200, 0),
    1: (150, 150, 150),
    2: (0, 0, 255),
    3: (255, 150, 0),
    4: (0, 220, 255),
}


def download_file(url: str, dest: Path) -> Path:
    """Descarga un archivo si no existe."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Descargando {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    """Extrae un zip si el directorio destino no existe."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    return out_dir


def build_data_yaml(dataset_dir: Path) -> Path:
    """Genera un data.yaml con rutas absolutas (evita problemas de rutas relativas)."""
    dataset_dir = Path(dataset_dir).resolve()
    cfg = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "valid/images",
        "names": CLASS_NAMES,
    }
    if (dataset_dir / "test" / "images").exists():
        cfg["test"] = "test/images"
    out = dataset_dir / "data_local.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def read_yolo_labels(label_path: Path):
    """Lee un .txt YOLO -> lista de (cls, xc, yc, w, h) normalizados."""
    boxes = []
    if Path(label_path).exists():
        for line in Path(label_path).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                boxes.append((int(parts[0]), *map(float, parts[1:5])))
    return boxes


def draw_boxes(image, boxes):
    """Dibuja bounding boxes YOLO normalizadas con su etiqueta sobre una imagen BGR."""
    img = image.copy()
    h, w = img.shape[:2]
    for cls, xc, yc, bw, bh in boxes:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES.get(cls, str(cls))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), color, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def visualize_samples(dataset_dir: Path, split: str = "train", n: int = 8,
                      cols: int = 4, seed: int = 42, save_path=None):
    """Muestra n imágenes aleatorias con sus cajas y opcionalmente guarda la figura."""
    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.Random(seed).shuffle(images)
    images = images[:n]

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, images):
        img = cv2.imread(str(p))
        boxes = read_yolo_labels(lbl_dir / f"{p.stem}.txt")
        ax.imshow(cv2.cvtColor(draw_boxes(img, boxes), cv2.COLOR_BGR2RGB))
        ax.set_title(p.name[:25], fontsize=8)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    return fig
