# AI Research on Wildfire Detection, Spread Forecasting & Geospatial Risk Mapping

**Researcher:** Muhammad Waheed Sabir ([GitHub: @waheedsabir](https://github.com/waheedsabir))  
**Status:** Prospective research planning and starter code (October 2026)  
**Scope:** Historical U.S. wildfire research; **not an operational warning system**

## Overview
This public repository records a prospective research direction exploring deep learning, computer vision and geospatial data analysis for wildfire monitoring. It contains a research protocol, source inventory, proposed milestones, and a simple exploratory NASA FIRMS CSV processing utility. The repository **does not yet report trained wildfire detection models, validated spread forecasts, or results from historical fire datasets**.

### Workstream 1: Fire and smoke detection
Proposed comparison of baseline YOLO-family object detection models with attention-enhanced candidates on appropriately labeled historical fire/smoke imagery. Planned metrics include precision, recall, F1, mAP@0.5 and smoke-versus-cloud false alarms.

### Workstream 2: Multimodal spread forecasting
Proposed investigation of time-aligned satellite observations, meteorology, vegetation/fuel indicators and terrain, comparing simple forecasting baselines with ConvLSTM or transformer-based fusion models where justified.

### Workstream 3: Geospatial interpretation
Proposed non-operational geospatial visualization of historical fire-related observations, uncertainty and risk context for research evaluation.

## Neural network training code

A **research-only** PyTorch image-level fire/no-fire training scaffold now supports ResNet18, MobileNetV3-Small, EfficientNet-B0 and optional single-node DistributedDataParallel (DDP). It has been exercised with synthetic images on CPU, including a two-process Gloo run; neither genuine wildfire data training nor multi-GPU execution has yet been verified. See [training instructions](training/README.md) and the [synthetic integration test](tests/test_training_pipeline.py). This classifier is distinct from the future YOLO fire/smoke *object detection* milestone.

## Contents
- [Prospective research protocol](RESEARCH_PROTOCOL.md)
- [Proposed implementation roadmap](ROADMAP.md)
- [Candidate public data sources](DATA_SOURCES.md)
- [Dated evidence and progress log](EVIDENCE_LOG.md)
- [FIRMS CSV descriptive utility](src/firms_hotspot_baseline.py)
- [Distributed deep learning training baseline (ResNet18, MobileNetV3, EfficientNet-B0)](training/README.md)
- [Synthetic unit test](tests/test_firms_hotspot_baseline.py)

## Reproduce the starter utility
Download an authentic CSV from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) and document the product, license and dates before using it:

```bash
python -m unittest discover -s tests -v
python src/firms_hotspot_baseline.py --input path/to/actual_firms_download.csv --out outputs/
```

This optional utility requires only Python's standard library. It outputs a JSON summary and GeoJSON *points*; these are **thermal anomaly observations, not fire polygons, smoke annotations, wildfire predictions, or risk estimates**. The included unit test uses artificial records purely to check software behavior. No actual FIRMS data is bundled.

## Scientific and authorship transparency
All workstreams are preliminary and subject to adjustment based on available labels, data coverage and empirical validation. Research claims will be accompanied by methods, code versions, dataset provenance and evaluation results when available. The initial project documentation and starter utility were prepared with AI assistance and should not be taken as independently authored peer-reviewed research or evidence of completed experiments. The named account owner is responsible for reviewing and adopting the work.

**Safety:** This repository is for research only. It must not be used for wildfire emergency alerts, evacuation guidance, or other safety-critical decisions.

## Record
- **October 2026:** Public repository created; prospective research and documentation starter materials added. Refer to actual GitHub commit timestamps for precise dates. Initial PyTorch classification/DDP software scaffolding added and tested on synthetic CPU images. No completed wildfire experiments or multi-GPU runs are claimed.
