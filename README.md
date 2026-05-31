# edge-lipsync-model

Clean training and evaluation pipeline for an edge-oriented Duix UNet lip-sync model.

Phase 1 keeps the current Duix UNet architecture unchanged, initializes from existing Duix
weights, and fine-tunes on supervised synchronized avatar videos.

Large assets are not committed. Keep raw videos, datasets, Wenet ONNX files, checkpoints, and
renders outside git and reference them through config files.
