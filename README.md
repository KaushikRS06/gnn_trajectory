# gnn_trajectory

## Environment setup
The Python environment of this project is managed with Conda. See [instructions/01_environment_setup.md](instructions/01_environment_setup.md) for the full instructions. In short:
```
conda env create -f conda-environment-files/conda-environment-cpu-<x>.yml
conda activate gnn_trajectory
```
, where `<x>` is either `unix` or `win`.

## Train the final models

From the project root, with the project environment activated:

```bash
python scripts/train_model.py --model attention
python scripts/train_model.py --model multimodal
```

Both use the existing 35 m datasets under `data/processed/` and the same fully
connected directed graph construction. Defaults: seed 42, 30 epochs, batch size
64, Adam learning rate 0.001. CUDA is used when available, otherwise CPU.

```bash
python scripts/train_model.py --model multimodal --seed 123 --epochs 30 --batch-size 64 --lr 0.001 --checkpoint-dir checkpoints
```

The best validation-loss checkpoint is reloaded for validation and test evaluation.
Checkpoints default to `checkpoints/{model}_seed{seed}_best.pth`. Repeating the same
model and seed in the same directory replaces that checkpoint; use another
`--checkpoint-dir` to preserve separate experiments. Metrics are printed to the
terminal. Random generators are seeded, but CUDA graph operations may still vary
across runs or hardware.

A sibling JSON manifest records the checkpoint's model, seed, radius, training
settings, best epoch, validation/test metrics, and PyTorch/PyG versions. It is
written after the best checkpoint is reloaded and evaluated.

Attention reports ADE/FDE. The six-mode model trains with best-of-K ADE and reports
minADE6/minFDE6, minimized independently across modes. Historical notebook FDE
results instead use the best-ADE mode and are not the same metric.

The notebooks retain explanatory code. Camera-inspection helpers live in `tools/`.
