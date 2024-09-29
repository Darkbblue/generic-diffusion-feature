# Semantic Segmentation Task
## Dataset
We use CityScapes and ADE20K datasets. Follow [the mmseg guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) to prepare the two datasets.

## Adding the model
You need to go to the installed mmseg lib at `anaconda3/envs/generic-diffusion-feature/lib/python3.9/site-packages/mmseg/models/segmentors`. Then copy the file `models/diffusion_segmentor.py` here. Also modify the `__init__.py` in the lib folder to import the newly added segmentor.

## Run
Below is a simple example:
```bash
python3 train.py configs/city_sdxl.py --work-dir /data/diffusion-feature/logs/city_sdxl
```
Change the config and working directory according to your own preference.  

You can find all available configs in `configs`:
- `ade_legacy_sd15` and `city_legacy_sd15`: Legacy-v1.5.
- `ade_legacy_sdxl` and `city_legacy_sdxl`: Legacy-XL.
- `ade_sd15` and `city_sd15`: Ours-v1.5.
- `ade_sdxl` and `city_sdxl`: Ours-XL.
- `ade_full` and `city_full`: Ours-XL-t. **This config requires two GPUs to run.**
- `ade_vpd` and `city_vpd`: Our implementation of [VPD](https://github.com/wl-zhao/VPD).

It's also possible to create your own configs and explore if there are even better feature combinations!  
