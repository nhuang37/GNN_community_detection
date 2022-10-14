# GNNs for community detection on sparse graphs

This repo contains implementations of (1) sampling from a Dense-Sparse-Graph-Model (DSGM); (2) running graph neural networks (GNNs) and spectral embeddings (SEs) on random graphs from DSGM; (3) compare GNNs and SEs on real-world graphs.

## Dependencies
- Python 3.7+
- Pytorch 1.10+
- [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

You can follow the code below to install pytorch-geometric
```
import os
import torch
os.environ['TORCH'] = torch.__version__
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```
## Experiments
- Simulation on DSGM (1),(2): ```Experiment_simulation.ipynb```
- Experiment on Amazon Photo network (3): ```Experiment_real_world.ipynb```
- Experiment results can be downloaded in ```result``` file
