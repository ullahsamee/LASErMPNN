# LASErMPNN: Small-Molecule Conditioned Protein Sequence Design

### Environment Setup

LASErMPNN can be run in inference-only mode in any environment with PyTorch, torch-scatter, and torch-cluster.


To install the training environment, run the following set of commands using a [MiniForge](https://github.com/conda-forge/miniforge/releases/tag/24.11.3-2) (recommended) or existing conda installation.

```bash
conda create --name laser_torch -y
conda activate laser_torch

conda install python=3.8 numpy pytorch pytorch-cuda=11.8 pytorch-scatter pytorch-cluster scipy -c pytorch -c nvidia -c pyg -c defaults --override-channels -y
pip install pykeops
conda install pandas matplotlib seaborn plotly jupyter scipy ProDy pytest scikit-learn h5py rdkit -c conda-forge -y
pip install logomaker wandb tqdm

```

You may need to update pytorch-cuda to your installed cuda version.

### Running Inference

### Running Batch Inference

### Training LASErMPNN

### Training Ligand Encoder


### Neural Iterative Selection & Expansion Implementation

# TODO:

- [ ] Release PDB Database to Zenodo and record DOIs.
- [ ] Make Github Release when all data uploaded (and create DOI?).
- [ ] 