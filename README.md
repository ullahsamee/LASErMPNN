# LASErMPNN: Small-Molecule Conditioned Protein Sequence Design

### Check out the paper [here](https://www.biorxiv.org/content/10.1101/2025.04.22.649862v1)!

![A block diagram of the LASErMPNN architecture depicting information flow through the network.](./images/laser_block_diagram.png)

> [!WARNING]
> A major difference between LASErMPNN and LigandMPNN is that LASErMPNN was trained on protonated structures. 
> Please make sure your ligand has the appropriate hydrogens (in the expected protonation state) attached when running the model or you may encounter unexpected behavior.
> For an example of how to adjust protonation from a smiles string, check out the script `./protonate_and_add_conect_records.py` from the [NISE repo](https://github.com/polizzilab/NISE/blob/main/protonate_and_add_conect_records.py)

### TODO:

- [ ] Make Release when finalized.
- [ ] Add Colab notebook.
- [ ] Add pdb code dataset split text files to avoid digging into code.
- [ ] Add LigandMPNN architecture retraining code.

### Environment Setup

A minimal version of LASErMPNN could be run in inference mode in any Python environment with PyTorch, torch-scatter, and torch-cluster. 
ProDy is used internally to read and write PDB files.

To install the training environment, run the following set of commands using a [MiniForge](https://github.com/conda-forge/miniforge/releases/tag/24.11.3-2) installation (recommended) or an existing conda installation with a libmamba solver.

```bash
mamba env create -f conda_env.yml -y
```

This will create an environment called `lasermpnn`.

To ensure your conda installation is using the libmamba solver, run `conda config --show-sources` 
and ensure the output has `solver: libmamba` at the bottom. 
If not, run `conda config --set-solver libmamba`.


### Running Inference

This script outputs a single pdb file named `laser_output.pdb` and is useful for testing:


```text
usage: python run_inference.py [-h] [--model_weights MODEL_WEIGHTS] [--output_path OUTPUT_PATH] [--temp SEQUENCE_TEMP] [--fs_sequence_temp FS_SEQUENCE_TEMP] [--bb_noise BACKBONE_NOISE] [--device DEVICE] [--fix_beta] [--ignore_statedict_mismatch]
                        [--ebd] [--repack_only] [--ignore_ligand]
                        input_pdb_code

Run LASErMPNN inference on a given PDB file.

positional arguments:
  input_pdb_code        Path to the input PDB file.

optional arguments:
  -h, --help            show this help message and exit
  --model_weights MODEL_WEIGHTS, -w MODEL_WEIGHTS
                        Path to dictionary of torch.save()ed model state_dict and training parameters. Default: ./model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt
  --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        Path to the output PDB file.
  --temp SEQUENCE_TEMP, -t SEQUENCE_TEMP
                        Sequence sample temperature.
  --fs_sequence_temp FS_SEQUENCE_TEMP, -f FS_SEQUENCE_TEMP
                        Residues around the ligand will be sampled at this temperature, otherwise they default to sequence_temp.
  --bb_noise BACKBONE_NOISE, -n BACKBONE_NOISE
                        Inference backbone noise.
  --device DEVICE, -d DEVICE
                        Pytorch style device string. Ex: "cuda:0" or "cpu".
  --fix_beta, -b        Residues with B-Factor of 1.0 have sequence and rotamer fixed, residues with B-Factor of 0.0 are designed.
  --ignore_statedict_mismatch, -s
                        Small state_dict mismatches are ignored. Don't use this unless any missing parameters aren't learned during training.
  --ebd, -e             Uses entropy based decoding order. Decodes all residues and selects the lowest entropy residue as next to decode, then recomputes all remaining residues. Takes longer than normal decoding.
  --repack_only         Only repack residues, do not design new ones.
  --ignore_ligand       Ignore ligands in the input PDB file.
```


### Running Batch Inference

This script is useful to generate multiple designs for one or multiple inputs. Creates an output directory with subdirectories for each input file (unless run with a single input file).

```text
usage: run_batch_inference.py [-h] [--designs_per_batch DESIGNS_PER_BATCH] [--model_weights_path MODEL_WEIGHTS_PATH] [--sequence_temp SEQUENCE_TEMP] [--first_shell_sequence_temp FIRST_SHELL_SEQUENCE_TEMP] [--chi_temp CHI_TEMP]
                              [--chi_min_p CHI_MIN_P] [--seq_min_p SEQ_MIN_P] [--device INFERENCE_DEVICE] [--use_water] [--silent] [--ignore_key_mismatch] [--disabled_residues DISABLED_RESIDUES] [--disable_charged_fs] [--fix_beta]
                              [--repack_only_input_sequence] [--ignore_ligand]
                              input_pdb_directory output_pdb_directory designs_per_input

Run batch LASErMPNN inference.

positional arguments:
  input_pdb_directory   Path to directory of input .pdb or .pdb.gz files, a single input .pdb or .pdb.gz file, or a .txt file of paths to input .pdb or .pdb.gz files.
  output_pdb_directory  Path to directory to output LASErMPNN designs.
  designs_per_input     Number of designs to generate per input.

optional arguments:
  -h, --help            show this help message and exit
  --designs_per_batch DESIGNS_PER_BATCH, -b DESIGNS_PER_BATCH
                        Number of designs to generate per batch. If designs_per_input > designs_per_batch, chunks up the inference calls in batches of this size. Default is 30, can increase/decrease depending on available GPU memory.
  --model_weights_path MODEL_WEIGHTS_PATH, -w MODEL_WEIGHTS_PATH
                        Path to model weights. Default: ./model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt
  --sequence_temp SEQUENCE_TEMP
                        Temperature for sequence sampling.
  --first_shell_sequence_temp FIRST_SHELL_SEQUENCE_TEMP
                        Temperature for first shell sequence sampling. Can be used to disentangle binding site temperature from global sequence temperature for harder folds.
  --chi_temp CHI_TEMP   Temperature for chi sampling.
  --chi_min_p CHI_MIN_P
                        Minimum probability for chi sampling. Not recommended.
  --seq_min_p SEQ_MIN_P
                        Minimum probability for sequence sampling. Not recommended.
  --device INFERENCE_DEVICE, -d INFERENCE_DEVICE
                        PyTorch style device string (e.g. "cuda:0").
  --use_water           Parses water (resname HOH) as part of a ligand.
  --silent              Silences all output except pbar.
  --ignore_key_mismatch
                        Allows mismatched keys in checkpoint statedict
  --disabled_residues DISABLED_RESIDUES
                        Residues to disable in sampling.
  --disable_charged_fs  Disables charged residues in the first shell.
  --fix_beta            If B-factors are set to 1, fixes the residue and rotamer, if not, designs that position.
  --repack_only_input_sequence
                        Repacks the input sequence without changing the sequence.
  --ignore_ligand       Ignore ligand in sampling.
```


### Training LASErMPNN

To retrain the model, download the datasets with `download_ligand_encoder_training_dataset.sh` and `download_protonated_pdb_training_dataset.sh` for each respective dataset by running them in the project's root directory.

We used 4x A6000 GPUs to train the LASErMPNN model which takes around 24 hrs for 60k optimizer steps. See `train_lasermpnn.py` for more information.


### Training Ligand Encoder

Training the Ligand Encoder module can be done with much lower memory and a single GPU. See `pretrain_ligand_encoder.py` for more information.


### Neural Iterative Selection & Expansion Implementation

see https://www.github.com/polizzilab/NISE for a NISE protocol implementation using Boltz-1.


