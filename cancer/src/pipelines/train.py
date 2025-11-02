"""
Pipeline de entrenamiento mínimo usando CancerDetectionModel.
Este script espera tensores preprocesados en NumPy (.npy) para un demo rápido.

Ejemplo:
  pwsh> python -m src.pipelines.train --x data/processed/X.npy --y data/processed/y.npy --model ResNet50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.config import load_config, configure_logging
from models.cancer_detection import CancerDetectionModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrenamiento mínimo de detección temprana")
    parser.add_argument("--x", type=str, required=False, help="Ruta a X.npy preprocesado")
    parser.add_argument("--y", type=str, required=False, help="Ruta a y.npy")
    parser.add_argument("--model", type=str, default="ResNet50", help="Tipo de modelo (ResNet50/ViT/Hybrid)")
    args = parser.parse_args()

    _ = load_config()
    logger = configure_logging()
    model = CancerDetectionModel()

    # Modo demo con datos aleatorios si no se proporcionan X/y
    if not args.x or not args.y:
        logger.warning("No se proporcionaron datos, generando demo sintética (no clínica)")
        rng = np.random.default_rng(42)
        X_train = rng.random((64, 224, 224, 3), dtype=np.float32)
        y_train = rng.integers(0, 2, size=(64, 1)).astype(np.float32)
        x_val = rng.random((16, 224, 224, 3), dtype=np.float32)
        y_val = rng.integers(0, 2, size=(16, 1)).astype(np.float32)
    else:
        X_train = np.load(args.x)
        y_train = np.load(args.y)
        # División simple para demo
        split = int(len(X_train) * 0.8)
        X_train, x_val = X_train[:split], X_train[split:]
        y_train, y_val = y_train[:split], y_train[split:]

    results = model.train_model((X_train, y_train), (x_val, y_val), model_type=args.model)
    print(results.get('validation_metrics', {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
