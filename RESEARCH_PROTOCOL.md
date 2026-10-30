# Prospective research protocol — draft

**Working title:** Reproducible AI Workflows for Wildfire Detection, Spread Forecasting and Geospatial Risk Mapping  
**Researcher/project owner:** Muhammad Waheed Sabir (GitHub: [@waheedsabir](https://github.com/waheedsabir))  
**Publication status:** Initial prospective protocol, published October 2026. **Not** a completed study, validated model, or peer-reviewed publication.

## Objective
Investigate reproducible computer-vision and multimodal-learning methods for historical U.S. wildfire observations, including fire/smoke detection, short-horizon fire-spread forecasting, and geospatial visualization. This document records **proposed work**, not a claim that the experiments have been completed.

## Research questions
- **RQ1 — Detection:** How do baseline object detectors perform on independently labeled fire/smoke imagery under varying weather and viewing conditions? How does attention modification affect false-positive rates?
- **RQ2 — Forecasting:** Can meteorology, satellite-derived observations, vegetation/fuel indices and terrain improve historical spread forecasts over persistence or other simple baselines?
- **RQ3 — Communication:** Can a research dashboard show historical risk layers, model uncertainty, data provenance and limitations clearly?

## Proposed design
1. Select historical U.S. wildfire events and document study periods and inclusion/exclusion criteria.
2. Inventory data sources, resolutions, time coverage, usage terms, missingness and annotation quality.
3. Differentiate active-fire hotspot *points* from image-level smoke/fire detection labels and independently mapped fire perimeters. These are not interchangeable labels.
4. Prepare reproducible splits by historical event and time to reduce leakage.
5. Evaluate baseline image detection (e.g. YOLO-family models); only then compare attention-enhanced variants such as ECA/CBAM with proper ablation tests.
6. Explore baseline and multimodal short-horizon forecasting, with chronological validation and uncertainty reporting.
7. Visualize only historical or research-grade outputs, not operational wildfire warnings.

## Planned evaluation
- Detection: precision, recall, F1, mAP@0.5, false positives, and per-condition errors on held-out data.
- Forecasting: spatial overlap/perimeter error at clearly defined forecast horizons, calibration and comparison against baseline methods.
- Reproducibility: exact code commit, dataset source/version/license, checksums, parameter settings, train/validation/test definitions, and failure cases.

## Limitations
Clouds, haze and glare can resemble smoke; satellite revisit intervals and geospatial resolutions vary; thermal anomaly detections are not exact fire boundaries; labels can be incomplete; historical model performance does not imply real-time emergency reliability.

## Planned deliverables
- Documented data inventory and exploratory hotspot-processing utility
- Reproducible detector experiments and error analysis
- Historical multimodal forecasting benchmark
- Non-operational geospatial prototype and technical report

## Research integrity and provenance
This is an **initial draft**. It does not claim an active deployed system, completed training, model accuracy, publications, grants, collaborations, or institutional endorsement. Any subsequent findings will be added with their actual completion dates and supporting evidence. Drafting assistance may include AI tools; final scientific claims must be verified by contributors.

**Not for operational emergency decisions.**