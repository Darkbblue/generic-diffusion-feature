# Semantic Correspondence Task

## Dataset
Download the dataset from http://cvlab.postech.ac.kr/research/SPair-71k/.  

Put the three json files in `dataset` to `SPair-71k/JPEGImages`. (They are taken from [diffusion_hyperfeatures](https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures).)  

## Run
Below is a simple example:
```bash
python3 task-corres.py \
    --log_path your/log/path \
    --dataset_path your/dataset/path/SPair-71k/JPEGImages \
    --configs correspondence/config_sdxl.json \
    --task_name sdxl \
    --algorithm conv
```

The available configs are:
- `config_legacy_sd15.json` for Legacy-v1.5.
- `config_legacy_sdxl.json` for Legacy-XL.
- `config_sd15.json` for Ours-v1.5.
- `config_sdxl.json` for Our-XL.

For Ours-XL-t, use the following command. This requires 4 GPUs to run.
```bash
python3 task-corres.py \
    --log_path your/log/path \
    --dataset_path your/dataset/path/SPair-71k/JPEGImages \
    --configs correspondence/config_sdxl.json correspondence/config_full_15.json correspondence/config_full_pgv2.json \
    --task_name full \
    --algorithm conv \
    --load_model_to_different_gpu
```
