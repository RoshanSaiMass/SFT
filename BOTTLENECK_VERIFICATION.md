# LOO bottleneck verification

Base: num_function 3e5556a. New backend only; existing backend files unchanged.

- 17 backend CPU tests passed: local training-only GELU reconstruction,
  all supported targeted adapter/tuner configurations, frozen-weight scope,
  LOO block choice, classifier training, test metrics, checkpoint round trips,
  failure detection and preservation of prior checkpoints.
- 13 existing num_function route and LOO integration tests passed.
- A100 launcher passed bash syntax validation, all 12 dataset ID dry runs,
  invalid-ID rejection, and explicit width 96 / DoRA rank 16 dry run.
- Actual full-size ViT-B/16 architecture instantiated without downloads for
  12 width/rank/adapter combinations; training scope asserted and parameters
  counted. The README reports those measured counts.
- No CUDA training, dataset accuracy benchmark, cluster scheduling, or claimed
  improvement over SFP was validated locally. Tests use a tiny ViT and synthetic
  images. Existing data splits, PEFT implementations, and budget defaults remain.
