model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='EfficientNetD3Backbone',  # Custom backbone
        pretrained=True
    ),
    neck=dict(
        type='BiFPN',
        in_channels=[48, 136, 384],
        out_channels=160,
        num_outs=5
    ),
rpn_head=dict(
    type='RepPointsHead',
    num_classes=80,
    in_channels=160,
    feat_channels=160,
    point_feat_channels=160,
    stacked_convs=3,
    num_points=9,
    gradient_mul=0.3,
    point_strides=[8, 16, 32, 64, 128],
    point_base_scale=4,
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0),
    loss_bbox_init=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.5),
    loss_bbox_refine=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
)
