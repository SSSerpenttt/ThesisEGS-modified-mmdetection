_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='EfficientDet',
        model_name='efficientdet-d3',
        pretrained=True
    ),
    neck=dict(
        type='BiFPN',
        num_levels=5,
        num_channels=160
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=80),
        mask_head=dict(num_classes=80)
    )
)
