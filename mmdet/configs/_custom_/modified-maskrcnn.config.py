default_scope = 'mmdet'

# Enhanced training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,  # Extended for better convergence
    val_interval=1,
    dynamic_intervals=[(20, 1)]  # More frequent late-stage validation
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# Model architecture using FCOS as anchor-free RPN
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='EfficientNet',
        arch='b3',
        out_indices=(2, 3, 4, 5),
        norm_cfg=dict(type='BN', requires_grad=True, momentum=0.1, eps=1e-3),
        norm_eval=False,
        frozen_stages=0
    ),
    neck=dict(
        type='BiFPN',
        in_channels=[32, 48, 136, 384],  # EfficientNet-B3 output dims
        out_channels=256,
        num_outs=4,
        stack=2,
        start_level=0,
        end_level=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        add_extra_convs='none'
    ),
    rpn_head=dict(
        type='RepPointsHead',
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64],
        point_base_scale=4,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=0.5),
        loss_bbox_refine=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64],
            finest_scale=56
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=17,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.5
            ),
            loss_bbox=dict(
                type='IoULoss',
                loss_weight=2.5
            )
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign', output_size=28, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64],
            finest_scale=56
        ),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=17,
            loss_mask=dict(
                type='DiceLoss',
                loss_weight=0.15,
                activate=True,
                eps=1e-5
            )
        )
    ),
    # Ensure these are defined elsewhere or added directly here
    train_cfg=dict(
        rpn=dict(
            init=dict(
                assigner=dict(
                    type='PointAssigner',
                    scale=4,
                    pos_num=1
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False
            ),
            refine=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    match_low_quality=True,
                    ignore_iof_thr=-1
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False
            )
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            mask_size=28,
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5
        )
    )
)


# Dataset configuration
data_root = '/content/group_egs_augmented/'
classes = (
    'damage-crack', 'damage-dent', 'damage-scratch', 'damage-tear',
    'panel-apillar', 'panel-bonnet', 'panel-bpillar',
    'panel-cpillar', 'panel-dpillar', 'panel-fender',
    'panel-frontdoor', 'panel-qpanel', 'panel-reardoor',
    'panel-rockerpanel', 'panel-roof', 'panel-tailgate', 'panel-trunklid'
)

metainfo = {
    'classes': classes,
    'palette': [
        (220, 20, 60),    # Red
        (0, 128, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 140, 0),    # Orange
        (255, 215, 0),    # Gold
        (128, 0, 128),    # Purple
        (255, 105, 180),  # Pink
        (64, 224, 208),   # Turquoise
        (123, 104, 238),  # Slate Blue
        (255, 69, 0),     # Orange Red
        (154, 205, 50),   # Yellow Green
        (70, 130, 180),   # Steel Blue
        (95, 158, 160),   # Cadet Blue
        (255, 20, 147),   # Deep Pink
        (0, 191, 255),    # Sky Blue
        (189, 183, 107),  # Khaki
        (160, 82, 45),    # Sienna
    ]
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)


train_dataloader = dict(
    batch_size=2,  # Will stack into (B, C, H, W)
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='/content/group_egs_augmented/train/_annotations.coco.json',
        data_prefix=dict(img='/content/group_egs_augmented/train/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')  # ✅ include 'pad_shape'
            )
        ]
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='/content/group_egs_augmented/valid/_annotations.coco.json',
        data_prefix=dict(img='/content/group_egs_augmented/valid/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')  # ✅ include 'pad_shape'
            )
        ]
    )
)

test_dataloader = val_dataloader.copy()
test_dataloader['dataset']['ann_file'] = '/content/group_egs_augmented/test/_annotations.coco.json'
test_dataloader['dataset']['data_prefix'] = dict(img='/content/group_egs_augmented/test/')
test_dataloader['collate_fn'] = dict(type='default_collate')  # ✅ Add this too

# Enhanced evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/content/group_egs_augmented/valid/_annotations.coco.json',
    metric=['bbox', 'segm'],
    classwise=True,
    # iou_thrs=[0.5, 0.75],  # Added stricter IoU threshold
    format_only=False
)

test_evaluator = val_evaluator.copy()
test_evaluator['ann_file'] = '/content/group_egs_augmented/test/_annotations.coco.json'

# Optimized training configuration
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
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
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=100,
        log_metric_by_epoch=True
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=50,
        max_keep_ckpts=3,  # Limit checkpoint storage
        save_best='auto',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='DetVisualizationHook',
        interval=25,
        draw=True
    ),
    early_stop=dict(
        type='EarlyStoppingHook',
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

log_level = 'DEBUG' #==> DEBUG, INFO
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
load_from = None
resume = False

# Mixed precision training
fp16 = dict(loss_scale=dict(init_scale=512), opt_level='O1')

# Visualization
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')  # Added TensorBoard
    ],
    name='visualizer'
)
