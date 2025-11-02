"""
Preparación E2E para NSCLC (o colección similar) con datos reales de TCIA:
- Descarga series DICOM (limitadas) y procesa a PNG (+ metadatos) -> labels_nsclc.csv
- Hace merge con CSV clínico externo (PatientID + columna de etiqueta) -> labels_nsclc_labeled.csv
- Extrae características radiómicas 2D (fallback skimage) -> radiomics_nsclc.csv
- Genera CSV final listo para entrenamiento multimodal -> train_nsclc.csv

Uso ejemplo:
  pwsh> python -m src.pipelines.nsclc_prepare --collection NSCLC-Radiomics --modality CT \
        --max-patients 5 --max-series 2 \
        --clinical-csv data/external/nsclc_clinical.csv --clinical-id-col PatientID --label-col Histology \
        --out-dir data/processed/nsclc
Luego entrenar:
  pwsh> python -m src.pipelines.train_fusion --labels_csv data/processed/nsclc/train_nsclc.csv \
        --image_col filepath --label_col label --epochs 15 --k_folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

# Asegurar que 'src' esté en sys.path cuando se ejecuta como script
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config_loader import configure_logging  # type: ignore
from utils.tcia_client import TCIAClient  # type: ignore
from utils.dicom_processor import DICOMProcessor  # type: ignore
from pipelines.extract_radiomics import compute_features  # type: ignore


def _collect_metadata(client: TCIAClient, processor: DICOMProcessor, *,
                      collection: str, modality: str | None, max_patients: int, max_series: int,
                      images_dir: Path) -> pd.DataFrame:
    patients_local = client.get_patients(collection)[: max_patients]
    rows_local = []
    for p in patients_local:
        pid = p.get("PatientID", "")
        series_list = client.get_series(collection, patient_id=pid)
        if modality:
            series_list = [s for s in series_list if s.get("Modality") == modality]
        series_list = series_list[: max_series]
        for s in series_list:
            suid = s.get("SeriesInstanceUID", "")
            if not suid:
                continue
            zip_path = client.download_series(suid)
            if not zip_path:
                continue
            target_dir = images_dir / suid
            target_dir.mkdir(parents=True, exist_ok=True)
            result = processor.process_dicom_series(zip_path, str(target_dir))
            for md in result.get("metadata", []):
                rows_local.append({
                    "filepath": md.get("processed_image_path"),
                    "PatientID": md.get("PatientID"),
                    "StudyInstanceUID": md.get("StudyInstanceUID"),
                    "SeriesInstanceUID": md.get("SeriesInstanceUID"),
                    "Modality": md.get("Modality"),
                    "BodyPartExamined": md.get("BodyPartExamined"),
                    "slice_index": md.get("slice_index"),
                })
    return pd.DataFrame(rows_local)


def _compute_radiomics(df_in: pd.DataFrame) -> pd.DataFrame:
    rows_local = []
    for _, row in df_in.iterrows():
        feats = compute_features(str(row["filepath"]))
        feats["filepath"] = row["filepath"]
        rows_local.append(feats)
    return pd.DataFrame(rows_local)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preparación E2E NSCLC con TCIA + clínico externo")
    parser.add_argument("--collection", type=str, default="NSCLC-Radiomics")
    parser.add_argument("--max-patients", type=int, default=5)
    parser.add_argument("--max-series", type=int, default=2)
    parser.add_argument("--modality", type=str, default="CT")
    parser.add_argument("--clinical-csv", type=str, required=True, help="CSV externo con columnas de paciente y etiqueta")
    parser.add_argument("--clinical-id-col", type=str, default="PatientID")
    parser.add_argument("--label-col", type=str, default="Histology")
    parser.add_argument("--out-dir", type=str, default="data/processed/nsclc")
    args = parser.parse_args()

    logger = configure_logging()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    client = TCIAClient()
    processor = DICOMProcessor()

    # 1) Descargar/Procesar -> labels_nsclc.csv con metadatos
    meta_df = _collect_metadata(client, processor,
                                collection=args.collection,
                                modality=args.modality,
                                max_patients=args.max_patients,
                                max_series=args.max_series,
                                images_dir=images_dir)
    labels_csv = out_dir / "labels_nsclc.csv"
    if meta_df.empty:
        logger.error("No se generaron imágenes/metadata. Ajuste parámetros.")
        return 1
    meta_df.to_csv(labels_csv, index=False)

    # 2) Merge con clínico -> labels_nsclc_labeled.csv
    clin_df = pd.read_csv(args.clinical_csv)
    if args.clinical_id_col not in clin_df.columns or args.label_col not in clin_df.columns:
        raise ValueError(f"El CSV clínico debe contener columnas {args.clinical_id_col} y {args.label_col}")
    merged = meta_df.merge(clin_df[[args.clinical_id_col, args.label_col]], left_on="PatientID", right_on=args.clinical_id_col, how="left")
    merged.rename(columns={args.label_col: "label"}, inplace=True)
    merged.drop(columns=[args.clinical_id_col], inplace=True)
    labeled_csv = out_dir / "labels_nsclc_labeled.csv"
    merged.to_csv(labeled_csv, index=False)

    # 3) Extraer radiomics (fallback) -> radiomics_nsclc.csv y merge final
    feats_df = _compute_radiomics(merged)
    feats_csv = out_dir / "radiomics_nsclc.csv"
    feats_df.to_csv(feats_csv, index=False)

    final_df = merged.merge(feats_df, on="filepath", how="left")
    train_csv = out_dir / "train_nsclc.csv"
    final_df.to_csv(train_csv, index=False)

    logger.info(f"Preparación completada. CSV final: {train_csv} con {len(final_df)} filas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
