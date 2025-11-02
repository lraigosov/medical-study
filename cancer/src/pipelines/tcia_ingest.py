"""
Ingesta real desde TCIA: descarga series DICOM, procesado a PNG y generación de CSV de metadatos/etiquetas.
Uso básico (ejemplos):
  pwsh> python -m src.pipelines.tcia_ingest --collection CMB-LCA --max-patients 3 --max-series 2 \
         --label-by BodyPartExamined --out-images data/processed/images --out-csv data/processed/labels.csv

Opcionalmente mapear etiquetas:
  --label-map '{"CHEST": "Pulmon", "BREAST": "Mama"}'
  o bien: --label-map-file config/label_map.json

Filtrado por modalidad:
  --modality CT  (o MR, PT, etc.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import pandas as pd

# Asegurar que 'src' esté en sys.path cuando se ejecuta como script
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import configure_logging  # type: ignore
from utils.tcia_client import TCIAClient  # type: ignore
from utils.dicom_processor import DICOMProcessor  # type: ignore


def _parse_label_map(args) -> Optional[Dict[str, str]]:
    if args.label_map:
        try:
            return json.loads(args.label_map)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"label_map inválido: {e}")
    if args.label_map_file:
        p = Path(args.label_map_file)
        if not p.exists():
            raise FileNotFoundError(f"No existe label_map_file: {p}")
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _series_for_patient(client: TCIAClient, collection: str, pid: str, modality: Optional[str], max_series: int):
    series_list = client.get_series(collection, patient_id=pid)
    if modality:
        series_list = [s for s in series_list if s.get("Modality") == modality]
    return series_list[: max_series]


def _rows_from_series(processor: DICOMProcessor, zip_path: str, target_dir: Path, label_by: Optional[str], label_map: Optional[Dict[str, str]]):
    rows: List[Dict[str, str]] = []
    result = processor.process_dicom_series(zip_path, str(target_dir))
    for md in result.get("metadata", []):
        md_fields = {
            "filepath": str(md.get("processed_image_path")),
            "PatientID": md.get("PatientID"),
            "StudyInstanceUID": md.get("StudyInstanceUID"),
            "SeriesInstanceUID": md.get("SeriesInstanceUID"),
            "Modality": md.get("Modality"),
            "BodyPartExamined": md.get("BodyPartExamined"),
            "slice_index": md.get("slice_index"),
        }
        label_value = None
        if label_by:
            source_val = md.get(label_by)
            if label_map and source_val in label_map:
                label_value = label_map[source_val]
            else:
                label_value = source_val
        md_fields["label"] = label_value
        rows.append(md_fields)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingesta TCIA a imágenes procesadas + CSV")
    parser.add_argument("--collection", type=str, required=True, help="Nombre de colección TCIA")
    parser.add_argument("--max-patients", type=int, default=5)
    parser.add_argument("--max-series", type=int, default=3)
    parser.add_argument("--modality", type=str, default=None, help="Filtrar por modalidad (CT/MR/PT/..)")
    parser.add_argument("--out-images", type=str, default="data/processed/images")
    parser.add_argument("--out-csv", type=str, default="data/processed/labels.csv")
    parser.add_argument("--label-by", type=str, default=None, help="Campo de metadatos para etiqueta (e.g., BodyPartExamined, Modality, SeriesDescription)")
    parser.add_argument("--label-map", type=str, default=None, help="JSON con mapeo valor->etiqueta")
    parser.add_argument("--label-map-file", type=str, default=None, help="Ruta a JSON con mapeo valor->etiqueta")
    args = parser.parse_args()

    logger = configure_logging()
    client = TCIAClient()
    processor = DICOMProcessor()
    label_map = _parse_label_map(args)

    out_images = Path(args.out_images)
    out_images.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Obtener pacientes y series
    patients = client.get_patients(args.collection)[: args.max_patients]
    rows: List[Dict[str, str]] = []

    for p in patients:
        pid = p.get("PatientID", "")
        for s in _series_for_patient(client, args.collection, pid, args.modality, args.max_series):
            series_uid = s.get("SeriesInstanceUID", "")
            if not series_uid:
                continue
            zip_path = client.download_series(series_uid)
            if not zip_path:
                continue
            target_dir = out_images / args.collection / series_uid
            target_dir.mkdir(parents=True, exist_ok=True)
            rows.extend(_rows_from_series(processor, zip_path, target_dir, args.label_by, label_map))

    if not rows:
        logger.error("No se generaron filas; verifique colección/modality/parametría")
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info(f"CSV generado: {out_csv} ({len(df)} filas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
