"""Reproducible PulsePoint dataset audit. Run: python src/audit.py"""
import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "reports"
OUT.mkdir(exist_ok=True)

def read(name):
    with (RAW / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def missing(rows, col):
    return sum(row.get(col, "") == "" for row in rows)

def duplicate_count(rows, keys):
    return sum(n - 1 for n in Counter(tuple(row[k] for k in keys) for row in rows).values() if n > 1)

def main():
    regional = read("noah_regional.csv")
    qc = read("noah_qc.csv")
    original = read("mendeley_original.csv")
    imputed = read("mendeley_imputed.csv")

    result = {
        "regional": {
            "rows": len(regional),
            "regions": len({r["REGION"] for r in regional}),
            "years": sorted({int(r["YR"]) for r in regional}),
            "duplicate_location_week_keys": duplicate_count(regional, ["REGION", "YR", "WN"]),
            "approximated_rainfall_rows": sum(r["FLAG_DEKADAL_APPROX"] == "1" for r in regional),
            "plausibility_flagged_rows": sum(r["FLAG_PLAUSIBILITY"] == "1" for r in regional),
            "missing_dengue_counts": missing(regional, "DC_DOH")
        },
        "quezon_city": {
            "rows": len(qc),
            "duplicate_week_keys": duplicate_count(qc, ["YR", "WN"]),
            "missing_nasa_rainfall": missing(qc, "RF_NASA"),
            "missing_pagasa_rainfall": missing(qc, "RF_PAGASA")
        },
        "mendeley": {
            "weekly_rows": len(original),
            "missing_original_leptospirosis": missing(original, "Leptospirosis"),
            "missing_imputed_leptospirosis": missing(imputed, "Leptospirosis"),
            "geographic_id_column_available": False,
            "year_cells_in_original_are_sparse": True
        }
    }
    target = OUT / "reproducible_audit.json"
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("Saved:", target)

if __name__ == "__main__":
    main()
