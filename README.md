# MeshVPR: Citywide Visual Place Recognition Using 3D Meshes

Official code for the paper ["MeshVPR: Citywide Visual Place Recognition Using 3D Meshes"](https://arxiv.org/abs/2406.02776) by Gabriele Berton, Lorenz Junglas, Riccardo Zaccone, Thomas Pollok, Barbara Caputo and Carlo Masone.

MeshVPR localizes real-world query images by doing VPR on a synthetic database of images, obtained from a 3D mesh of a city.

<p float="left">
  <img src="https://github.com/gmberton/gmberton.github.io/blob/96db6aa36b20d3a07f03e889ef82ec74370991cf/assets/MeshVPR/berlin_short_01.gif" width="40%" />
  <img src="https://github.com/gmberton/gmberton.github.io/blob/96db6aa36b20d3a07f03e889ef82ec74370991cf/assets/MeshVPR/berlin_short_03.gif" width="40%" />
  <img src="https://github.com/gmberton/gmberton.github.io/blob/96db6aa36b20d3a07f03e889ef82ec74370991cf/assets/MeshVPR/sf_short_01.gif" width="40%" />
  <img src="https://github.com/gmberton/gmberton.github.io/blob/96db6aa36b20d3a07f03e889ef82ec74370991cf/assets/MeshVPR/sf_short_02.gif" width="40%" />
</p>

## Quick Setup

You can start using MeshVPR by running these commands
```
git clone https://github.com/gmberton/MeshVPR
cd MeshVPR

mkdir data && cd data
# Download 100k images for training
rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/mesh_vpr_datasets/train_set_hq_100000 .
mkdir test_sets && cd test_sets
# Download val set and Melbourne test set
rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/mesh_vpr_datasets/test_sets/val_set .
rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/mesh_vpr_datasets/test_sets/synt_melbourne .

cd ../..

python train.py \
    --real_train_dir data/train_set_hq_100000/real_database \
    --synt_train_dir data/train_set_hq_100000/synt_database \
    --test_dir data/test_sets \
    --method cosplace
```

## Datasets and reproducibility

We provide all datasets to fully replicate our results. See Table 1 of the paper for further information on each dataset. For convenience, the training datasets come already paired (i.e. real and synt sets with precisely matching images).

| Type  |   Name    | Size  | Directory                |
| ----- | -------   |------ |------                    |
| Test  | Berlin    | 41 GB | test_sets/synt_berlin    |
| Test  | Paris     | 45 GB | test_sets/synt_paris     |
| Test  |Melbourne  | 11 GB | test_sets/synt_melbourne |
| Test  | Val       | 4.1 GB| test_sets/val_set        |
| Train | SF-HQ     | 670 GB| train_set_hq         |
| Train | SF-LQ     | 258 GB| train_set_lq         |
| Train | SF-HQ 1M  | 73 GB | train_set_hq_1000000 |
| Train | SF-HQ 100k| 7.4 GB| train_set_hq_100000  |
| Train | SF-HQ 10k | 748 MB| train_set_hq_10000   |

You can download any subset using `rsync` like this

`rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/mesh_vpr_datasets/DIRECTORY .`

for example

`rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/mesh_vpr_datasets/test_sets/synt_berlin .`
