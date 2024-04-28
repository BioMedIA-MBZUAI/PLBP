# PLBP: Protein-Ligand Binding Prediction through Diffusion Models
This repository is the official implementation of PLBP: Protein-Ligand Binding Prediction through Diffusion Models, inspired by _DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design._


## Dependencies
### Install via Conda and Pip
```bash
conda create -n decompdiff_new python=3.8
conda activate decompdiff_new
pip install numpy==1.20.0
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --force-reinstall pyg_lib==0.3.0 torch_scatter==2.1.0 torch_sparse==0.6.16 torch_cluster==1.6.0 torch_spline_conv==1.2.1 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install torch_geometric
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge

# For decomposition
conda install -c conda-forge mdtraj
pip install alphaspace2

# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr 
conda install -c conda-forge vina==1.2.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

## Preprocess 
```bash
python scripts/data/preparation/preprocess_subcomplex.py configs/preprocessing/crossdocked.yml
```
The processed dataset file is provided [here](https://drive.google.com/drive/folders/1z74dKcDKQbwpo8Uf8EJpGi12T4GCD8_Z?usp=share_link).

## Training
To train the model from scratch, you need to download the *.lmdb, *_name2id.pt and split_by_name.pt files and put them in the _data_ directory. Then, you can run the following command:
```bash
python scripts/train_diffusion_decomp.py configs/training.yml

# To continue training from the last checkpoint
python scripts/train_diffusion_decomp_edited.py configs/training.yml --ckpt_path #Last checkpoint path
```

## Sampling
To sample molecules given protein pockets in the test set, you need to download test_index.pkl and test_set.zip files, unzip it and put them in the _data_ directory. Then, you can run the following command:
```bash
python scripts/sample_diffusion_decomp.py configs/sampling_drift.yml  \
  --outdir $SAMPLE_OUT_DIR -i $DATA_ID --prior_mode {ref_prior, beta_prior}
```
If you want to sample molecules with beta priors, you also need to download files in this [directory](https://drive.google.com/drive/folders/1QOQOuDxdKkipYygZU9OIQUXqV9C28J5O?usp=share_link).


## Evaluation
```bash
python scripts/evaluate_mol_from_meta_full.py $SAMPLE_OUT_DIR \
  --docking_mode {none, vina_score, vina_full} \
  --aggregate_meta True --result_path $EVAL_OUT_DIR
```


