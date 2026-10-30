# Public research progress and evidence log

**2026-10 (initial public repository setup):** Prospective research plan, data-source inventory, and starter descriptive processing utility published. This establishes the existence of *public planning material and starter code*, not completed research experiments, historic research activity, institutional support, or validated results.

Future entries should be dated on the actual activity date and link directly to the relevant commits, issues, releases, data manifests, or independent evidence.

| Actual date (UTC) | Activity completed | Direct evidence | Limitations |
| --- | --- | --- | --- |
| Initial GitHub publication | Prospective documentation and starter utility | See repository commits | No model training or wildfire research results reported |

**Evidence policy:** Do not backdate entries or characterize a plan or synthetic test as field validation. Preserve raw training logs, unsuccessful experiments, third-party feedback, and actual data provenance when those exist.
**2026-10-30 (software baseline):** Added AI-assisted PyTorch image-classification training scaffold supporting ResNet18, MobileNetV3-Small, EfficientNet-B0 and optional DDP. Developer-only synthetic-image CPU smoke tests (one-process and two-process/Gloo) were executed in the assistant's working environment. These were software execution tests, not wildfire-data training, validated field performance, GPU scaling results, or evidence of original authorship. See [training README](training/README.md) and [test code](tests/test_training_pipeline.py).
