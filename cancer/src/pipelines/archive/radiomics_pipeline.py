"""
Pipeline sencillo para extracción radiómica a partir de imágenes y máscaras.

Ejemplo:
  pwsh> python -m src.pipelines.radiomics_pipeline --images data/processed/*.png --out results/radiomics_features.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.config_loader import load_config, configure_logging
from analysis.radiomics_analysis import RadiomicsAnalyzer


def main() -> int:
    parser = argparse.ArgumentParser(description="Extracción radiómica masiva")
    parser.add_argument("--images", type=str, required=True, help="Patrón glob de imágenes")
    parser.add_argument("--out", type=str, default="results/radiomics_features.json", help="Ruta de salida")
    args = parser.parse_args()

    _ = load_config()
    logger = configure_logging()
    analyzer = RadiomicsAnalyzer()

    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        logger.error("No se encontraron imágenes para procesar")
        return 1

    df = analyzer.extract_features_batch(image_paths)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="table")
    logger.info(f"Características guardadas en {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
