from __future__ import annotations

import argparse
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)
    if "slice_index" in df.columns:
        df["label"] = (df["slice_index"] % 2).map({0: "Benigno", 1: "Maligno"})
    else:
        # Si no hay slice_index, alternar por fila
        df = df.reset_index(drop=True)
        df["label"] = (df.index % 2).map({0: "Benigno", 1: "Maligno"})
    df.to_csv(args.out_csv, index=False)
    print(f"Escribí {args.out_csv} con distribución: {df['label'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
