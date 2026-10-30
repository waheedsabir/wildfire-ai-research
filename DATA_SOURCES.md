# Prospective data sources and provenance

These sources are **candidates**, not evidence that datasets have been downloaded or analyzed.

| Candidate | Official link | Intended use | Important limitation |
| --- | --- | --- | --- |
| NASA FIRMS | https://firms.modaps.eosdis.nasa.gov/download/ | Explore historical MODIS/VIIRS thermal anomaly points | Hotspots are not smoke bounding boxes or precise wildfire perimeters |
| USGS MTBS | https://www.mtbs.gov/ | Historical burned-area and burn-severity context | Coverage, temporal alignment and resolution require study-specific checks |
| USGS EarthExplorer | https://earthexplorer.usgs.gov/ | Investigate suitable imagery | Licenses, acquisition dates, cloud cover and band availability must be verified |
| NOAA NCEI | https://www.ncei.noaa.gov/ | Candidate historical meteorological data | Stations/grids and time alignment must be validated |

## Dataset record template
For any dataset actually used, record its source URL, exact product, version, access date (UTC), geographical and date coverage, usage terms, preprocessing, missing data, SHA-256 checksum (when downloadable), and how the source was used.

No real-world wildfire model validation should be inferred from the synthetic unit test in this repository.