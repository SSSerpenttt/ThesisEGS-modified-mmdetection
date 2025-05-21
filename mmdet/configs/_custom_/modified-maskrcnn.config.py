# Top-level training loop config (must be named train_cfg for MMEngine)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Model internal train_cfg and test_cfg (for heads, etc)
model_train_cfg = dict(
    rpn=dict(
        assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
        sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False
    ),
    rcnn=dict(
        assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1),
        sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False
    )
)

model_test_cfg = dict(
    rpn=dict(
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0
    ),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

model = dict(
    type='mmdet.MaskRCNN',
    backbone=dict(
        type='mmdet.EfficientNet',
        arch='b3',
        out_indices=(1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://efficientnet_b3'),
    ),
    neck=dict(
        type='mmdet.BIFPN',
        in_channels=[48, 136, 384],
        out_channels=256,
        num_outs=5,
        stack=2
        # activation=dict(type='Swish'),
    ),
    rpn_head = dict(
        type='RepPointsRPNHead',
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        transform_method='minmax',
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
    ),
    roi_head=dict(
        type='mmdet.StandardRoIHead',
        bbox_roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='mmdet.RoIAlign',
                output_size=7,
                sampling_ratio=2
            ),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]
        ),
        bbox_head=dict(
            type='mmdet.Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=17
        ),
        mask_roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='mmdet.RoIAlign',
                output_size=14,
                sampling_ratio=2
            ),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]
        ),
        mask_head=dict(
            type='mmdet.FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=17
        )
    ),
    train_cfg=model_train_cfg,
    test_cfg=model_test_cfg
)

# Dataset root
data_root = '/content/group-egs_modified_pipeline/'

# Custom classes
classes = (
    'damage-crack', 'damage-dent', 'damage-scratch', 'damage-tear',
    'panel-apillar', 'panel-bonnet', 'panel-bpillar',
    'panel-cpillar', 'panel-dpillar', 'panel-fender',
    'panel-frontdoor', 'panel-qpanel', 'panel-reardoor',
    'panel-rockerpanel', 'panel-roof', 'panel-tailgate', 'panel-trunklid'
)

# Visualization colors (palette)
metainfo = {
    'classes': classes,
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 128, 0), (0, 0, 255),
        (255, 140, 0), (255, 215, 0), (128, 0, 128), (255, 105, 180),
        (64, 224, 208), (123, 104, 238), (255, 69, 0), (154, 205, 50),
        (70, 130, 180), (95, 158, 160), (255, 20, 147), (0, 191, 255),
        (189, 183, 107)
    ]
}

# Dataloaders
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        ann_file=data_root + 'train/_annotations.coco.json',
        data_prefix=dict(img=data_root + 'train/'),
        filter_cfg=dict(filter_empty_gt=True),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        ann_file=data_root + 'valid/_annotations.coco.json',
        data_prefix=dict(img=data_root + 'valid/')
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        ann_file=data_root + 'test/_annotations.coco.json',
        data_prefix=dict(img=data_root + 'test/')
    )
)

# Evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric=['bbox', 'segm']
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric=['bbox', 'segm']
)

# Optimizer and LR schedule
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
]

# Runtime settings
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

log_level = 'INFO'
load_from = 'checkpoints/retinanet_effb3_fpn_crop896_8x4_1x_coco_20220322_234806-615a0dda.pth'
resume = False

# Enable mixed precision training
fp16 = dict(loss_scale='dynamic')
