_base_ = [
    '_base_/models/fpn_r50.py', '_base_/datasets/cityscapes.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]
crop_size = (1024, 1024)
# normalize to [-1, 1]
# i know it may not look like [-1, 1] but it works as intended
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[v * 255 for v in [0.5, 0.5, 0.5]],
    std=[v * 255 for v in [0.5, 0.5, 0.5]],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=1,
    size=crop_size)

head_c=512
c_per_level = [2560, 640, 320]  # modify this for different settings
model = dict(
    type='DiffusionSegmentor',
    neck=dict(
        type='FPN',
        in_channels=c_per_level,
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        num_classes=19,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512,512)),
    data_preprocessor=data_preprocessor,

    # modify the follwing for different settings
    diffusion_feature=dict(
        layer={'up-level1-repeat1-vit-block0-cross-q': True, 'up-level1-repeat2-res-out': True,
        'up-level2-repeat1-vit-block0-cross-q': True, 'up-level3-repeat0-vit-block0-self-k': True,},
        version='1-5',
        device='cuda',
        attention=None,
        img_size=512,
        t=50,
        train_unet=True,  # you can turn this off to reduce VRAM usage
    ),
    feature_layers=[
        [('up-level1-repeat1-vit-block0-cross-q', 1280), ('up-level1-repeat2-res-out', 1280)],
        [('up-level2-repeat1-vit-block0-cross-q', 640)],
        [('up-level3-repeat0-vit-block0-self-k', 320)],
    ],
    prompt='An urban street scene with multiple lanes, various buildings, traffic lights, cars in the lanes, and pedestrians, highly realistic.',
    prompt_tuning=True,
)

optimizer = dict(type='AdamW', lr=0.00016, weight_decay=0.001)
optim_wrapper = dict(
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'unet': dict(lr_mult=0.1),
            'norm': dict(decay_mult=0.),
        }
    ),
    # accumulative_counts=8,
)
lr_config = dict(
    policy='poly', power=1, min_lr=0.0, by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6
)
# we modify optimizer directly in _base_/schedules instead
# if you want to change lr / weight_decay, uncomment the two lines above

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
# train_dataloader = dict(batch_size=8)  # set this if you want to change batch size
