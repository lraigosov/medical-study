from __future__ import annotations

import argparse
from pathlib import Path

from ..infrastructure.container import build_container


def main() -> int:
    parser = argparse.ArgumentParser(description="Análisis de imagen médica con IA (Gemini)")
    parser.add_argument("image", type=str, help="Ruta a la imagen a analizar")
    parser.add_argument("--type", dest="atype", type=str, default="general", help="Tipo de análisis")
    parser.add_argument("--config", type=str, default=None, help="Ruta a config.json (opcional)")
    args = parser.parse_args()

    container = build_container(args.config)
    svc = container.analysis_service
    result = svc.analyze_image(args.image, args.atype)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
