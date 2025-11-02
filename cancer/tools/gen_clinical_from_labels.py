from __future__ import annotations

import argparse
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="Genera CSV clínico sintético (PatientID, Histology) desde labels.csv")
    p.add_argument("--in_labels", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.in_labels)
    if "PatientID" not in df.columns:
        raise SystemExit("labels.csv no contiene columna PatientID")
    u = df[["PatientID"]].drop_duplicates().reset_index(drop=True)
    # Alterna dos valores de histología para balancear mínimamente
    choices = ["Adenocarcinoma", "SquamousCell"]
    u["Histology"] = [choices[i % 2] for i in range(len(u))]
    u.to_csv(args.out_csv, index=False)
    print(f"Escribí {args.out_csv} con {len(u)} pacientes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
