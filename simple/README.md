# SIMPLE
Code repository for the SIGMOD 24 paper:
"SIMPLE: Efficient Temporal Graph Neural Network Training at Scale with Dynamic Data Placement"
## Requirements
- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511
- numba 0.54.1

Compile C++ temporal sampler (from TGL) first with the following command
> python SIMPLE/setup.py build_ext --inplace

## Datasets
We use four datasets in the paper: LASTFM, WIKITALK, STACKOVERFLOW and GDELT.
For LASTFM and GDELT, they can be downloaded from AWS S3 bucket using the `down.sh` script. 
For WIKITALK and STACKOVERFLOW, they can be downloaded from http://snap.stanford.edu/data/wiki-talk-temporal.html and https://snap.stanford.edu/data/sx-stackoverflow.html respectively.
Note that for WIKITALK and STACKOVERFLOW, they need to be preprocessed after obtaining the raw data from the links above. For example:
> python preprocess.py --data \<NameOfDataset> --txt \<PathOfRawData>

python preprocess.py --data LASTFM --txt /raid/guorui/DG/dataset/LASTFM

## Usage
To generate buffer plans by SIMPLE, run:
> python SIMPLE/buffer_plan_preprocessing.py --data \<NameOfDataset> --config \<TrainingConfiguration> --dim_edge_feat \<EdgeFeatDimension> --dim_node_feat \<NodeFeatDimension> --mem_dim \<MemoryDimension> --threshold \<UserDefinedBudget>

python SIMPLE/buffer_plan_preprocessing.py --data LASTFM --config config/ORCA_LASTFM.yml
python SIMPLE/buffer_plan_preprocessing.py --data MOOC --config config/ORCA_REDDIT.yml

python SIMPLE/buffer_plan_preprocessing.py --data LASTFM --config config/TGN_LASTFM.yml
python SIMPLE/buffer_plan_preprocessing.py --data REDDIT --config config/TGN_REDDIT.yml
python SIMPLE/buffer_plan_preprocessing.py --data LASTFM --config config/TGN_LASTFM.yml
python SIMPLE/buffer_plan_preprocessing.py --data GDELT --config config/TGN_GDELT.yml

Exemplar training:
> python main.py --data WIKITALK --config config/TGN_WIKITALK.yml --gpu 0 --threshold 0.1

python main.py --data LASTFM --config config/ORCA_LASTFM.yml --gpu 0 --threshold 0.1
python main.py --data LASTFM --config config/TGN_LASTFM.yml --gpu 0 --threshold 0.1
python main.py --data GDELT --config config/TGN_GDELT.yml --gpu 0 --threshold 0.1
python main.py --data TALK --config config/TGN_GDELT.yml --gpu 0 --threshold 0.1