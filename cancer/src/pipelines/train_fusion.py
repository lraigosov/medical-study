"""
Entrenamiento del modelo de fusión (imagen + radiómico opcional) con K-Fold CV.
Uso:
  pwsh> python -m src.pipelines.train_fusion --images_dir data/processed/images --labels_csv data/processed/labels.csv --image_col filepath --label_col label
Si no se proporciona labels_csv, se asume estructura por carpetas: images_dir/{benigno,maligno,...}/*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

import pandas as pd

# Asegurar que 'src' esté en sys.path cuando se ejecuta como script
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold

from utils.config_loader import configure_logging  # type: ignore
from models.fusion_model import FusionConfig, FusionCancerModel  # type: ignore


def _infer_from_folders(images_dir: Path, exts=(".png", ".jpg", ".jpeg")) -> pd.DataFrame:
    rows = []
    for sub in images_dir.iterdir():
        if not sub.is_dir():
            continue
        label = sub.name
        for p in sub.rglob('*'):
            if p.suffix.lower() in exts:
                rows.append({"filepath": str(p), "label": label})
    return pd.DataFrame(rows)


essential_cols = ["filepath", "label"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrenar modelo de fusión multimodal")
    parser.add_argument("--images_dir", type=str, required=False, default="data/processed/images", help="Directorio con imágenes")
    parser.add_argument("--labels_csv", type=str, required=False, help="CSV con filepath,label y columnas radiómicas opcionales")
    parser.add_argument("--image_col", type=str, default="filepath")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--radiomics_cols", type=str, nargs="*", default=None, help="Nombres de columnas radiómicas")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k_folds", type=int, default=5)
    args = parser.parse_args()

    logger = configure_logging()
    images_dir = Path(args.images_dir)

    if args.labels_csv:
        df = pd.read_csv(args.labels_csv)
        if args.image_col not in df.columns or args.label_col not in df.columns:
            raise ValueError(f"labels_csv debe contener columnas {args.image_col} y {args.label_col}")
    else:
        df = _infer_from_folders(images_dir)
        if df.empty:
            raise ValueError("No se encontraron imágenes. Proporcione --labels_csv o estructura por carpetas.")

    # Normalizar rutas
    df[args.image_col] = df[args.image_col].astype(str)

    # Configuración del modelo
    cfg = FusionConfig(epochs=args.epochs, batch_size=args.batch_size, k_folds=args.k_folds)
    model = FusionCancerModel(cfg)

    # Radiomics opcional
    radiomics_cols: Optional[List[str]] = args.radiomics_cols if args.radiomics_cols else None

    summary = model.fit_kfold(
        df=df,
        image_col=args.image_col,
        label_col=args.label_col,
        radiomics_cols=radiomics_cols,
    )

    out_path = model.run_dir / "summary_min.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_val_accuracy": summary.get("best_val_accuracy"),
            "best_model_path": summary.get("best_model_path"),
        }, f, indent=2)

    logger.info(f"Entrenamiento finalizado. Mejor modelo: {summary.get('best_model_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
