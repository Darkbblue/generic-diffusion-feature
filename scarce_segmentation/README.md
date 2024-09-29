# Label-Scarce Segmentation Task

This task requires very large RAM (not VRAM), as all features are stored in RAM.  

## Dataset
Download the Horse-21 dataset from https://github.com/yandex-research/ddpm-segmentation.

## Run
For this task, we separately extract features and later load them for discrimination.

### Feature
Use the standalone feature extraction script in the codebase for this step:

```bash
python3 extract_feature.py \
    --version xl \
    --img_size 1024 \
    --t 50 \
    --layer feature/configs/config_xl_practical.json \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "your/dataset/path/horse_21/real/test/*.png" \
    --output_dir your/feature/path/horse_21/sdxl/ \
    --split test
```
Set the input dir and split as `test` for the testing set, and set as `train` for the training set.
Set image size as 512 for SDv1.5 and 1024 for SDXL.
The prompt used is `a photo of a horse`.  

All available settings:
- `config_15_legacy.json` for Legacy-v1.5.
- `config_xl_legacy.json` for Legacy-XL.
- `config_15_practical.json` for Ours-v1.5.
- `config_xl_practical.json` for Ours-XL and as part of Ours-XL-t.
- `config_pg_amalgamation.json` as part of Ours-XL-t.
- `config_15_amalgamation.json` as part of Ours-XL-t. Set `--attention up_cross` along with this config.

### Discrimination
```bash
python3 task-pixel.py \
    --category horse_21 \
    --dataset_path your/dataset/path/horse_21/real/ \
    --log_path your/log/path \
    --feature_path your/feature/path/horse_21 \
    --feature_id sdxl \
    --feature_len 3840 \
    --task_name sdxl \
    --batch_size 64 \
    --shuffle_dataset
```

Note that this requires manually setting the number of channels as `--feature_len`. The channel numbers of each setting are:
- Legacy-v1.5: 3520
- Legacy-XL: 2240
- Ours-v1.5: 3520
- Ours-XL: 3840
- Ours-XL-t: 8154

The results reported in the paper are the average of five random runs, where each run randomly split the dataset into training and testing sets. This shuffling is done in the discrimination script, so you don't need to re-extract features for each run.  

### Ours-XL-t
This setting might be a bit complicated, so we detail it here.

First extract features: (Also remember to run for the training set.)
```bash
python3 extract_feature.py \
    --version xl \
    --img_size 1024 \
    --t 50 \
    --layer feature/configs/config_xl_practical.json \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "your/dataset/path/horse_21/real/test/*.png" \
    --output_dir your/feature/path/horse_21/sdxl/ \
    --split test

python3 extract_feature.py \
    --version 1-5 \
    --img_size 512 \
    --t 50 \
    --layer feature/configs/config_15_amalgamation.json \
    --attention up_cross \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "your/dataset/path/horse_21/real/test/*.png" \
    --output_dir your/feature/path/horse_21/sd15_amal/ \
    --split test

python3 extract_feature.py \
    --version pgv2 \
    --img_size 1024 \
    --t 50 \
    --layer feature/configs/config_pg_amalgamation.json \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "your/dataset/path/horse_21/real/test/*.png" \
    --output_dir your/feature/path/horse_21/pg_amal/ \
    --split test
```

Then do the discrimination:
```bash
python3 task-pixel.py \
    --category horse_21 \
    --dataset_path your/dataset/path/horse_21/real/ \
    --log_path your/log/path \
    --feature_path your/feature/path/horse_21 \
    --feature_id sdxl sd15_amal pg_amal \
    --feature_len 8154 \
    --task_name full \
    --batch_size 64 \
    --shuffle_dataset
```
