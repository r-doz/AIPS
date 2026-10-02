# Gulf map context for notebook 61

`gisco_2024_01m_gulf_countries.geojson` is a small local subset of
[Eurostat GISCO Countries 2024, 1:1 million](https://gisco-services.ec.europa.eu/distribution/v2/countries/countries-2024-units.html),
covering Italy, Slovenia and Croatia around the Gulf.

**© EuroGeographics for the administrative boundaries**

[GISCO administrative-unit use conditions](https://ec.europa.eu/eurostat/web/gisco/geodata/administrative-units)
permit non-commercial use with source acknowledgement. Include the credit above in the
map legend and the introductory page of a publication using these data. The notebook
includes it on figures and in exported captions. This is generalized geographic context,
not a surveyed coast or a maritime jurisdiction map.

`sources.json` records the exact source URLs, SHA-256 checksums, clip extent and processing.
To rebuild: download the three recorded country GeoJSON files, load with GeoPandas,
add English `ADMIN` labels Italy/Slovenia/Croatia, retain country code and geometry,
convert to EPSG:4326, concatenate and `geopandas.clip` to the recorded bounding box.
No simplification is applied. The notebook clips the local file further to the visible
map extent. It draws shared country boundaries on land only; no maritime borders are inferred.

`retained_grid_centres.csv` contains the 49 original longitude/latitude pairs shared by
`data/processed/cpr_gfw_2024.parquet` and `cpr_gfw_2025.parquet`. Extract unique centres
from each file and assert they match to regenerate it. This small snapshot permits
plotting without reading the full model dataset. Notebook 61 matches inverse-transformed
saved prediction positions to these centres with a strict 0.0001-degree tolerance and
checks one-to-one cell coverage per day. It never changes the saved counts.

The display support is the union of retained grid-cell footprints (half a grid spacing
around each original centre), intersected with water and the plot extent. It is **not**
a coastline or a maritime boundary. Unretained cells are never coloured, even where
the existing interpolation helper fills them temporarily with zeros. Interpolated
fields are masked after interpolation and clipped again to the vector support.
The original hand-traced Gulf polygon is not used as geographic context or as a mask.


## Maritime context overlay

`marine_regions_gulf_boundaries.geojson` contains three selected bilateral lines
from [Marine Regions / VLIZ Maritime Boundaries v12](https://www.marineregions.org/downloads.php),
under [CC BY 4.0](https://www.marineregions.org/disclaimer.php). `maritime_sources.json`
records the WFS request, original-response checksum, feature IDs, processing and source links.
To regenerate, issue the recorded request, select IDs 3511/3512/3513, and clip to the
GIS asset extent. The notebook clips these lines to visible water; coordinates are
not smoothed or extended to invent missing boundary segments. This is a derived local
research excerpt; refer to Marine Regions for the complete and current dataset.

Italy–Slovenia and Italy–Croatia use the source's treaty lines. Slovenia–Croatia uses
the source's 2017 arbitration line, styled separately and explicitly marked as disputed
by Croatia. The joint-regime perimeter and straight baselines are omitted because they
are not additional sovereign-country divisions. See the linked PCA award and Croatian
position in the metadata. These context lines do not alter model support or values.


`marine_regions_slovenian_waters.geojson` is the source dataset's Slovenia polygon
(MRGID 5692), retrieved from `MarineRegions:eez` through the WFS request recorded in
`maritime_sources.json`. The display-only mask excludes this area from all heatmap
panels when `mask_slovenian_waters=True`. Counts retain the dark sea background;
residuals retain their white sea background. All original cell values remain in
memory and exports. The polygon follows the same source representation of the
2017 arbitration line as the maritime overlay; it is not a statement of agreed jurisdiction.


`marine_regions_italian_waters.geojson` is the Marine Regions Italian EEZ polygon (MRGID 5682), clipped to the same local extent; see `italian_edge_smoothing_mask` in `maritime_sources.json` for its request and checksum. It restricts optional inward opacity feathering to Italian waters. Source: https://www.marineregions.org/eezdetails.php?mrgid=5682 (VLIZ; CC BY 4.0). No observations/predictions are changed or extrapolated by this display effect.

`marine_regions_croatian_waters.geojson` is the Marine Regions Croatian EEZ polygon (MRGID 5673), clipped to the same local extent for display exclusion. Request/checksum: `croatian_display_mask` in `maritime_sources.json`. Source: https://www.marineregions.org/eezdetails.php?mrgid=5673 (VLIZ; CC BY 4.0). Optional feathering now also fades at the Italian bilateral sea borders, leaving a 100 m clear display margin; this does not move the geographic lines.
