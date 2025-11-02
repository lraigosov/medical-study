"""
Pipeline de EDA rápida sobre colecciones TCIA.

Uso:
  pwsh> python -m src.pipelines.eda --collection CMB-LCA --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.config import load_config, configure_logging
from utils.tcia_client import TCIAClient


def main() -> int:
    parser = argparse.ArgumentParser(description="EDA rápida de colecciones TCIA")
    parser.add_argument("--collection", type=str, required=False, help="Nombre de colección TCIA")
    parser.add_argument("--limit", type=int, default=10, help="Límite de pacientes a listar")
    args = parser.parse_args()

    logger = configure_logging()
    client = TCIAClient()

    if not args.collection:
        cols = client.get_collections()
        logger.info(f"Colecciones disponibles: {len(cols)}")
        printed = min(10, len(cols))
        print(json.dumps(cols[:printed], indent=2))
        return printed

    # EDA de una colección
    patients = client.get_patients(args.collection)[: args.limit]
    studies = client.get_studies(args.collection)
    series = client.get_series(args.collection)

    summary = {
        "collection": args.collection,
        "patients_sample": patients,
        "n_patients": len(patients),
        "n_studies": len(studies),
        "n_series": len(series),
    }
    print(json.dumps(summary, indent=2))
    return summary.get("n_series", 0)


if __name__ == "__main__":
    raise SystemExit(main())
