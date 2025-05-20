_base_ = [
    '../_base_/models/mask-rcnn_efficientdet_d3_bifpn.py',  # Replace ResNet-101 + FPN
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
