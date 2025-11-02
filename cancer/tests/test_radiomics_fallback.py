import os
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipelines.extract_radiomics import compute_features


def test_compute_features_on_synthetic_image(tmp_path: Path):
    # Crear imagen sintética 64x64 con patrón simple
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[16:48, 16:48] = 200
    img_path = tmp_path / "synthetic.png"
    Image.fromarray(arr).save(img_path)

    feats = compute_features(str(img_path))
    # Debe contener algunas claves principales de cada grupo de features
    assert "int_mean" in feats
    assert "glcm_contrast_mean" in feats
    assert "edge_ratio" in feats
