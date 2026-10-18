# Selected source fields

| Source | Column | Meaning |
|---|---|---|
| NOAH Regional | REGION, YR, WN | Regional unit, year, surveillance week |
| NOAH Regional | DC_DOH | Dengue cases |
| NOAH Regional | RF_HDX | Rainfall series; all rows flagged as approximated from dekadal intervals |
| NOAH Regional | Temp_ERA5-Land, SH_ERA5-Land | Temperature, specific humidity |
| NOAH Regional | FLAG_DEKADAL_APPROX | Approximation provenance; 1 on all records |
| NOAH Regional | FLAG_PLAUSIBILITY | Source plausibility flag |
| NOAH QC | DC_QC, RF_NASA, RF_PAGASA | City dengue, NASA rainfall, PAGASA rainfall |
| Mendeley | Year, Week | Year is populated at each year boundary and must be filled down |
| Mendeley | Dengue, Leptospirosis | Disease counts, original sheet contains missing values |
| Mendeley | geographic unit | **Not present:** confirm national scope from publisher and metadata |

Do not use raw totals as comparable incidence without population denominators.
The NOAH source workbook contains a richer data dictionary; consult it for units.
