### Training Databases.

- Download by running `download_ligand_encoder_training_dataset.sh` and `download_protonated_pdb_trianing_dataset.sh` in the root directory.

- `./spice_dataset` contains SPICE dataset for ligand encoder pretraining.

- `./pdb_dataset` contains our protonated processed PDB database stored in a python [shelve](https://docs.python.org/3/library/shelve.html) object. See `./utils/pdb_dataset.py` for how to work with the data. 

