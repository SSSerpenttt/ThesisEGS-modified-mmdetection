# Enhanced training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,  # Extended for better convergence
    val_interval=1,
    dynamic_intervals=[(20, 1)]  # More frequent late-stage validation
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimized model training configuration
model_train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='mmdet.MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1
        ),
        sampler=dict(
            type='mmdet.RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=True
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    rcnn=dict(
        assigner=dict(
            type='mmdet.MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,  # Adjusted from 0.3 for better precision
            min_pos_iou=0.5,
            match_low_quality=False,  # Stricter matching for RCNN
            ignore_iof_thr=-1
        ),
        sampler=dict(
            type='mmdet.RandomSampler',
            num=512,
            pos_fraction=0.35,  # Increased positive samples
            neg_pos_ub=-1,
            add_gt_as_proposals=True
        ),
        pos_weight=-1,
        debug=False,
        mask_size=56
    )
)

# Enhanced test configuration
model_test_cfg = dict(
    rpn=dict(
        score_thr=0.01,
        nms_pre=3000,  # Increased from 300
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0
    ),
    rcnn=dict(
        score_thr=0.3,
        nms=dict(type='soft_nms', iou_threshold=0.5),  # Changed to soft_nms
        max_per_img=200,  # Increased from 100
        mask_thr_binary=0.45  # Adjusted threshold
    )
)

# Model architecture
model = dict(
    type='mmdet.MaskRCNN',
    backbone=dict(
        type='mmdet.EfficientNet',
        arch='b3',
        out_indices=(3, 4, 5),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint='https://download.openmmlab.com/mmdetection/v2.0/efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco/retinanet_effb3_fpn_crop896_8x4_1x_coco_20220322_234806-615a0dda.pth'
        ),
        norm_cfg=dict(type='BN', requires_grad=True)  # Added BN config
    ),
    neck=dict(
        type='mmdet.BiFPN',
        in_channels=[48, 136, 384],
        out_channels=256,
        num_outs=5,
        stack=2,
        start_level=0,
        end_level=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        add_extra_convs='on_output'
    ),
    rpn_head=dict(
        type='mmdet.RepPointsRPNHead',
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        stacked_convs=3,
        num_classes=1,
        transform_method='minmax',
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox_init=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0,
            loss_weight=0.5
        ),
        loss_bbox_refine=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0,
            loss_weight=1.0
        ),
        train_cfg=dict(
            init=dict(
                assigner=dict(
                    type='mmdet.PointAssigner',
                    scale=32,
                    pos_num=5
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
                grad_clip=dict(max_norm=35, norm_type=2)
            ),
            refine=dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1
                ),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=512,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False
                ),
                allowed_border=0,
                pos_weight=-1,
                debug=False
            )
        ),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=1000
        )
    ),
    roi_head=dict(
        type='mmdet.StandardRoIHead',
        bbox_roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=2
            ),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128],
            finest_scale=56  # Added for better small object handling
        ),
        bbox_head=dict(
            type='mmdet.Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=19,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=False,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0
            ),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0,
                loss_weight=1.0
            )
        ),
        mask_roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=28,
                sampling_ratio=2
            ),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128],
            finest_scale=56  # Added for better small object handling
        ),
        mask_head=dict(
            type='mmdet.FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=19,
            loss_mask=dict(
                type='mmdet.DiceLoss',
                loss_weight=2.0,
                activate=False,
                eps=1e-5
            )
        )
    ),
    train_cfg=model_train_cfg,
    test_cfg=model_test_cfg
)

# Dataset configuration
data_root = '/content/GroupEGS-Thesis-Dataset/'
classes = (
    'damage-crack', 'damage-dent', 'damage-scratch', 'damage-tear',
    'panel-apillar', 'panel-bonnet', 'panel-bpillar',
    'panel-cpillar', 'panel-dpillar', 'panel-fender',
    'panel-frontdoor', 'panel-qpanel', 'panel-reardoor',
    'panel-rockerpanel', 'panel-roof', 'panel-tailgate', 'panel-trunklid',
    'depth-deep', 'depth-shallow'
)

metainfo = {
    'classes': classes,
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 128, 0), (0, 0, 255),
        (255, 140, 0), (255, 215, 0), (128, 0, 128), (255, 105, 180),
        (64, 224, 208), (123, 104, 238), (255, 69, 0), (154, 205, 50),
        (70, 130, 180), (95, 158, 160), (255, 20, 147), (0, 191, 255),
        (189, 183, 107), (0, 255, 127), (160, 82, 45)
    ]
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# Enhanced data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='mmdet.RandomChoiceResize',
        scales=[(896, 896), (800, 800), (1024, 1024)],
        keep_ratio=True
    ),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),  # More conservative range
        saturation_range=(0.8, 1.2),
        hue_delta=18
    ),
    dict(
        type='mmdet.Albu',
        transforms=[
            dict(type='RandomRotate90', p=0.5),
            dict(type='Cutout', num_holes=8, max_h_size=32, max_w_size=32, p=0.5)
        ]
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,  # Increased from 1 if GPU memory allows
    num_workers=4,  # Increased from 2
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='mmdet.DefaultSampler', shuffle=True),
    dataset=dict(
        type='mmdet.CocoDataset',
        metainfo=metainfo,
        ann_file=data_root + 'train/_annotations_cleaned_final.coco.json',
        data_prefix=dict(img=data_root + 'train/'),
        filter_cfg=dict(
            filter_empty_gt=True,
            min_size=16  # Filter very small objects
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='mmdet.DefaultSampler', shuffle=False),
    dataset=dict(
        type='mmdet.CocoDataset',
        metainfo=metainfo,
        ann_file=data_root + 'valid/_annotations_cleaned_final.coco.json',
        data_prefix=dict(img=data_root + 'valid/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='mmdet.Resize', scale=(896, 896), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=32),
            dict(type='mmdet.PackDetInputs')
        ]
    )
)

test_dataloader = val_dataloader.copy()
test_dataloader['dataset']['ann_file'] = data_root + 'test/_annotations_cleaned_final.coco.json'
test_dataloader['dataset']['data_prefix'] = dict(img=data_root + 'test/')

# Enhanced evaluation
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'valid/_annotations_cleaned_final.coco.json',
    metric=['bbox', 'segm'],
    classwise=True,
    iou_thrs=[0.5, 0.75],  # Added stricter IoU threshold
    format_only=False
)

test_evaluator = val_evaluator.copy()
test_evaluator['ann_file'] = data_root + 'test/_annotations_cleaned_final.coco.json'

# Optimized training configuration
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0002,
        weight_decay=0.05,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'neck': dict(lr_mult=0.5),
            'rpn_head': dict(lr_mult=1.5),
            'roi_head': dict(lr_mult=1.0)
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],  # Adjusted milestones
        gamma=0.1
    )
]

# Enhanced runtime configuration
default_hooks = dict(
    timer=dict(type='mmdet.IterTimerHook'),
    logger=dict(
        type='mmdet.LoggerHook',
        interval=50,
        log_metric_by_epoch=True
    ),
    param_scheduler=dict(type='mmdet.ParamSchedulerHook'),
    checkpoint=dict(
        type='mmdet.CheckpointHook',
        interval=1,
        max_keep_ckpts=3,  # Limit checkpoint storage
        save_best='auto'
    ),
    sampler_seed=dict(type='mmdet.DistSamplerSeedHook'),
    visualization=dict(
        type='mmdet.DetVisualizationHook',
        interval=10,
        draw=True
    ),
    early_stop=dict(
        type='mmdet.EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=5,
        min_delta=0.005
    )
)

env_cfg = dict(
    cudnn_benchmark=True,  # Enabled for fixed input size
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
    gc_collect_threshold=0.7  # Better memory management
)

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
load_from = None
resume = False

# Mixed precision training
fp16 = dict(loss_scale=dict(init_scale=512), opt_level='O1')

# Visualization
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')  # Added TensorBoard
    ],
    name='visualizer'
)
