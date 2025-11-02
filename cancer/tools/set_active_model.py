from __future__ import annotations

from pathlib import Path
import json


def latest_summary() -> Path | None:
    base = Path(__file__).resolve().parents[1] / "results" / "models"
    if not base.exists():
        return None
    cands = list(base.rglob("summary_min.json"))
    if not cands:
        cands = list(base.rglob("training_summary.json"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def main() -> int:
    s = latest_summary()
    if not s:
        print("No se encontró summary_min.json o training_summary.json")
        return 1
    data = json.loads(s.read_text(encoding="utf-8"))
    best = data.get("best_model_path")
    if not best:
        print("Summary no contiene best_model_path")
        return 1
    active = s.parent.parent / "active_model.txt"
    active.write_text(best, encoding="utf-8")
    print(f"Modelo activo: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
