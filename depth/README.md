# Depth Estimation Task

## Dataset

## Run
```bash
python3 depth/main.py --train_backbone --output_dir outputs/depth/hunyuan --batch_size 1 --config configs/config_hunyuan_selection1.json --data_path nyu_depth_v2/data_converted
```

Note that, by default, `train_backbone` is recommended for depth estimation.
