This repo contains code to reproduce the paper: *Neural sampling from cognitive maps enables goal-directed imagination and planning*.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for Python and dependency management:

```bash
uv sync
```

To include development tools, use `uv sync --group dev`.

## Experiments

The original notebooks have been converted into executable, Hydra-configured scripts. Run them from the repository root:

```bash
uv run python scripts/gcml_abstract_graph.py
uv run python scripts/gcml_grid_cell.py
uv run python scripts/gcml_tiling.py
```

Hydra overrides make runs reproducible without editing source code. For example:

```bash
uv run python scripts/gcml_abstract_graph.py epochs=10 n_nodes=16
uv run python scripts/gcml_grid_cell.py n_steps=1000 n_cells=100
uv run python scripts/gcml_tiling.py train=true epochs=5 batch_size=256
```

Configuration lives in `configs/`, and the tiling experiment reads its dataset from `dataset/` by default. Hydra creates a timestamped directory for every run under `outputs/YYYY-MM-DD/HH-MM-SS_experiment/`. Models and Hydra metadata are written there; figures are saved as numbered PNG files in its `images/` subdirectory. No interactive plot windows are opened.

Run the tests with:

```bash
uv run pytest
```

## Citation

@article{lin2025neural,
  title={Neural sampling from cognitive maps supports goal-directed planning and imagination},
  author={Lin, Hui and Yang, Yukun and Zhao, Rong and Pezzulo, Giovanni and Maass, Wolfgang},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
