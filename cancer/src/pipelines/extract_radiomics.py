"""
Extracción de características radiómicas 2D (fallback) desde imágenes PNG/JPG usando scikit-image.
- Toma un CSV de entrada (labels) con columna de ruta a imagen.
- Genera un CSV con features por fila y mergea con las columnas originales opcionalmente.
Uso:
  pwsh> python -m src.pipelines.extract_radiomics --labels-csv data/processed/labels.csv \
         --image-col filepath --out-csv data/processed/radiomics.csv --merge True
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from skimage import io, color, filters, feature, util
from skimage.feature import graycomatrix, graycoprops
from scipy import stats


def _safe_read_gray(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    # rgb2gray -> float64 [0,1]; convertir a uint8 para GLCM/entropy
    if img.dtype != np.uint8:
        img_u8 = util.img_as_ubyte(img)
    else:
        img_u8 = img
    return img_u8


def _glcm_features(img_u8: np.ndarray) -> Dict[str, float]:
    # Matriz de co-ocurrencia para distancias y ángulos típicos
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_u8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = {
        'contrast': graycoprops(glcm, 'contrast'),
        'correlation': graycoprops(glcm, 'correlation'),
        'energy': graycoprops(glcm, 'energy'),
        'homogeneity': graycoprops(glcm, 'homogeneity'),
        'ASM': graycoprops(glcm, 'ASM'),
        'dissimilarity': graycoprops(glcm, 'dissimilarity'),
    }
    # Promediar sobre distancias y ángulos
    out = {}
    for k, v in props.items():
        out[f'glcm_{k}_mean'] = float(np.mean(v))
        out[f'glcm_{k}_std'] = float(np.std(v))
    return out


def _intensity_features(img_u8: np.ndarray) -> Dict[str, float]:
    hist, _ = np.histogram(img_u8, bins=256, range=(0, 255), density=True)
    mean = float(np.mean(img_u8))
    std = float(np.std(img_u8))
    skew = float(stats.skew(img_u8.reshape(-1)))
    kurt = float(stats.kurtosis(img_u8.reshape(-1)))
    ent = float(stats.entropy(hist + 1e-12))
    med = float(np.median(img_u8))
    p10 = float(np.percentile(img_u8, 10))
    p90 = float(np.percentile(img_u8, 90))
    return {
        'int_mean': mean,
        'int_std': std,
        'int_median': med,
        'int_p10': p10,
        'int_p90': p90,
        'int_skew': skew,
        'int_kurt': kurt,
        'int_entropy_hist': ent,
    }


def _edges_texture(img_u8: np.ndarray) -> Dict[str, float]:
    # Canny edges
    edges = feature.canny(img_u8.astype(float) / 255.0, sigma=1.0)
    edge_ratio = float(np.mean(edges))
    # Sobel magnitude
    sob = filters.sobel(img_u8.astype(float))
    return {
        'edge_ratio': edge_ratio,
        'sobel_mean': float(np.mean(sob)),
        'sobel_std': float(np.std(sob)),
    }


def compute_features(path: str) -> Dict[str, Any]:
    try:
        img_u8 = _safe_read_gray(path)
        feats = {}
        feats.update(_intensity_features(img_u8))
        feats.update(_glcm_features(img_u8))
        feats.update(_edges_texture(img_u8))
        return feats
    except Exception as e:  # noqa: BLE001
        # Devuelve información de error manteniendo tipo compatible con merge posterior
        return {"_error": str(e)}


essential_cols = ["filepath"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extracción radiómica 2D (fallback)")
    parser.add_argument("--labels-csv", type=str, required=True)
    parser.add_argument("--image-col", type=str, default="filepath")
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--merge", type=str, default="True", help="Si 'True', adjunta las columnas originales al CSV de salida")
    args = parser.parse_args()

    df = pd.read_csv(args.labels_csv)
    if args.image_col not in df.columns:
        raise ValueError(f"No se encuentra la columna de imagen: {args.image_col}")

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        path = str(row[args.image_col])
        feats = compute_features(path)
        feats["filepath"] = path
        rows.append(feats)

    fdf = pd.DataFrame(rows)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.merge.strip().lower() == "true":
        m = pd.merge(df, fdf, on="filepath", how="left")
        m.to_csv(out_path, index=False)
    else:
        fdf.to_csv(out_path, index=False)

    print(f"Features guardados en: {out_path} ({len(fdf)} filas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
