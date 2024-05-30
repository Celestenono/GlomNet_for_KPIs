# GlomNet: A HoVer Deep Learning Model for Glomerulus Instance Segmentation
Accepted to ISBI 2024
Read the paper [here](glomnet.pdf)

Code based on the monai implementation of the EfficientUNet and the HoVerNet

## Usage

1) Fill the cross validation file with the names of your WSIs
2) Modify the paths in [training_conf_hover.py](training_config_hover.yml), [training_conf_bin.py](training_config_bin.yml) [inf_conf_hover.py](inf_config_hover.yml), [inf_conf_bin.py](inf_config_bin.yml)
3) In [run_training.py](source/training/run_training.py) and [inference.py](source/inference/inference.py)  select the correct config file
4) Train with `python source/deep_learning_seg/training/run_training.py`
5) Infer with `python source/deep_learning_seg/inference/inference.py`

## Models checkpoints
Models weights to come

## Notes
The code will be available after the ISBI 2024 conference
