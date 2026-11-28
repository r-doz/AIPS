# Ebro Delta vs Trieste data audit (2025)

## Comparability checks

- GFW effort version: Ebro ['public-global-fishing-effort:v4.0']; Trieste ['public-global-fishing-effort:v4.0'].
- AIS grouping: Ebro `FLAGANDGEARTYPE`; Trieste `VESSEL_ID`. Absolute AIS counts are not strictly like-for-like because the observation units differ.
- Grid size: Ebro 86 cells; Trieste 49 cells.
- Days: Ebro 365/365; Trieste 365/365.
- The geometry plot shows Copernicus grid centers and the configured bounding boxes. It does not show coastlines or independently validate a marine mask.

## Region metrics

| Metric | Ebro Delta | Trieste |
|---|---:|---:|
| Rows | 31390 | 17885 |
| Grid cells | 86 | 49 |
| Duplicate date-cell keys | 0 | 0 |
| Cells outside configured bbox | 0 | 0 |
| AIS group-by | FLAGANDGEARTYPE | VESSEL_ID |
| AIS total | 13128 | 4818 |
| AIS count per cell-day | 0.4182 | 0.2694 |
| AIS active cell-day rate | 0.15 | 0.1075 |
| Satellite total | 640 | 4276 |

## Files

- `summary.json`: full metrics and validation results.
- `geometry_comparison.png`: grid centers and bounding boxes.
- `activity_comparison.png`: monthly normalized activity overview.
