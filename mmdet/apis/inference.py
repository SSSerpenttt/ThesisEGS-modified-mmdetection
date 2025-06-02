# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.dataset import default_collate
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet.registry import DATASETS
from mmdet.utils import ConfigType
from ..evaluation import get_classes
from ..registry import MODELS
from ..structures import DetDataSample, SampleList
from ..utils import get_test_pipeline_cfg

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

import torch
torch.autograd.set_detect_anomaly(True)

from mmengine.registry import TRANSFORMS
import numpy as np

from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.structures.mask import PolygonMasks
from mmdet.structures.mask import mask2bbox

# Place this after you get predictions and ground truths for an image
import matplotlib.pyplot as plt
import numpy as np

def show_image_with_boxes_and_masks(img, pred_bboxes, pred_masks, gt_bboxes, gt_masks):
    plt.figure(figsize=(12, 6))
    # Show image
    plt.imshow(img.transpose(1, 2, 0).astype(np.uint8))
    # Plot predicted bboxes
    for box in pred_bboxes:
        x1, y1, x2, y2 = box[:4]
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2))
    # Overlay predicted masks
    if pred_masks is not None:
        for mask in pred_masks:
            plt.imshow(mask, alpha=0.3, cmap='Reds')
    # Plot GT bboxes
    for box in gt_bboxes:
        x1, y1, x2, y2 = box[:4]
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=2))
    # Overlay GT masks
    if gt_masks is not None:
        for mask in gt_masks:
            plt.imshow(mask, alpha=0.3, cmap='Greens')
    plt.axis('off')
    plt.show()

@TRANSFORMS.register_module()
class RenameGtLabels:
    def __call__(self, results):
        # If "gt_labels" is missing and "gt_bboxes_labels" exists, rename it.
        if 'gt_bboxes_labels' in results and 'gt_labels' not in results:
            results['gt_labels'] = results['gt_bboxes_labels']
        return results

@TRANSFORMS.register_module()
class InspectAnnotations:
    def __call__(self, results):
        # print("InspectAnnotations running", flush=True)
        # Check for ground-truth boxes:
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            # If the boxes are wrapped (e.g., in a HorizontalBoxes),
            # try to get the underlying tensor.
            if hasattr(gt_bboxes, 'tensor'):
                bbox_tensor = gt_bboxes.tensor
            else:
                bbox_tensor = gt_bboxes
        #     print("BBoxes shape:", np.array(bbox_tensor.cpu().numpy()).shape, flush=True)
        # else:
        #     print("No gt_bboxes found!", flush=True)
        
        # Check for ground-truth labels:
        if 'gt_bboxes_labels' in results:
            gt_labels = results['gt_bboxes_labels']
            # It might be a list of labels or a tensor.
            if isinstance(gt_labels, torch.Tensor):
                labels_array = gt_labels.cpu().numpy()
            else:
                labels_array = np.array(gt_labels)
        #     print("Labels shape:", labels_array.shape, flush=True)
        # else:
        #     print("No gt_bboxes_labels found!", flush=True)
        
        # Optionally, check gt_masks if available:
        if 'gt_masks' in results:
            gt_masks = results['gt_masks']
        #     # Depending on the format, this might be a custom object.
        #     # If using PolygonMasks, you might check the number of masks.
        #     if hasattr(gt_masks, 'masks'):
        #         print("gt_masks has", len(gt_masks.masks), "masks", flush=True)
        #     else:
        #         print("gt_masks:", type(gt_masks), flush=True)
        # else:
        #     print("No gt_masks found!", flush=True)
        
        return results

@TRANSFORMS.register_module()
class EfficientNetPreprocessor:
    """Complete preprocessing pipeline for EfficientNet with batch dimension"""
    def __init__(self, size_divisor=32, mean=None, std=None, to_rgb=True):
        self.size_divisor = size_divisor
        self.mean = torch.tensor(mean).view(3, 1, 1) if mean else None
        self.std = torch.tensor(std).view(3, 1, 1) if std else None
        self.to_rgb = to_rgb
        
    def __call__(self, results):
        # 1. Ensure contiguous numpy array
        img = results['img']
        if isinstance(img, np.ndarray):
            img = np.ascontiguousarray(img)
        
        # 2. Convert to tensor
        img = torch.from_numpy(img).float()
        
        # 3. Handle channel order (RGB/BGR)
        if img.shape[2] == 3:  # HWC to CHW
            img = img.permute(2, 0, 1)
            if self.to_rgb:
                img = img[[2, 1, 0]]  # BGR to RGB
        
        # 4. Apply normalization
        if self.mean is not None and self.std is not None:
            img = (img - self.mean) / self.std
        
        # 5. Padding
        h, w = img.shape[-2], img.shape[-1]
        pad_h = (self.size_divisor - h % self.size_divisor) % self.size_divisor
        pad_w = (self.size_divisor - w % self.size_divisor) % self.size_divisor
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
        
        # 6. Store processed image (ensure contiguous)
        results['img'] = img.contiguous()
        results['img_shape'] = (img.shape[-2], img.shape[-1])
        return results

@TRANSFORMS.register_module()
class TensorPackDetInputs:
    def __init__(self, meta_keys=()):
        self.meta_keys = meta_keys

    def __call__(self, results):
        packed_results = {}

        # Pack image
        if 'img' in results:
            img = results['img']
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(np.ascontiguousarray(img))
            packed_results['inputs'] = img.contiguous()

        # Create DetDataSample
        data_sample = DetDataSample()
        gt_instances = InstanceData()

        # Handle bboxes
        if 'gt_bboxes' in results:
            raw_bboxes = results['gt_bboxes']
            if hasattr(raw_bboxes, 'tensor'):
                gt_instances.bboxes = raw_bboxes.tensor.float()
            else:
                gt_instances.bboxes = torch.as_tensor(raw_bboxes, dtype=torch.float32)
        
        # Handle labels - ensure we have same number as bboxes
        if 'gt_labels' in results:
            labels = torch.as_tensor(results['gt_labels'], dtype=torch.long)
        elif 'gt_bboxes_labels' in results:
            labels = torch.as_tensor(results['gt_bboxes_labels'], dtype=torch.long)
        else:
            num_boxes = len(gt_instances.bboxes) if hasattr(gt_instances, 'bboxes') else 0
            labels = torch.zeros(num_boxes, dtype=torch.long)

        
        gt_instances.labels = labels

        # Handle masks if present
        if 'gt_masks' in results:
            polygons = results['gt_masks']
            height, width = results['img_shape']
            formatted_polygons = []
            for seg in polygons:
                formatted_seg = [np.array(p, dtype=np.float32) for p in seg]
                formatted_polygons.append(formatted_seg)
            gt_instances.masks = PolygonMasks(formatted_polygons, height, width)

        data_sample.gt_instances = gt_instances

        # Pack metainfo
        meta_dict = {}
        for key in self.meta_keys:
            if key in results:
                meta_dict[key] = results[key]
        data_sample.set_metainfo(meta_dict)

        packed_results['data_samples'] = [data_sample]

        # if hasattr(gt_instances, 'bboxes'):
        #     print("[TensorPackDetInputs] GT bboxes (first 5):", gt_instances.bboxes[:5])
        # if hasattr(gt_instances, 'labels'):
        #     print("[TensorPackDetInputs] GT labels (first 5):", gt_instances.labels[:5])
        # if hasattr(gt_instances, 'masks'):
        #     print("[TensorPackDetInputs] GT masks (first 1):", type(gt_instances.masks), "Num:", len(gt_instances.masks) if hasattr(gt_instances.masks, '__len__') else 'N/A')

        return packed_results


@TRANSFORMS.register_module()
class DebugInput:
    def __call__(self, results):
        img = results['img']
        print(f"Final input - Shape: {img.shape}", flush=True)
        print(f"Channel means: {img.mean(dim=[1,2])}", flush=True)
        print(f"Keys in results: {list(results.keys())}", flush=True)
        print(f"Meta info preview: {[k for k in ['img_id', 'img_shape', 'ori_shape', 'scale_factor'] if k in results]}", flush=True)
        return results

def init_detector(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = 'none',
    device: str = 'cuda:0',
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to none.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    scope = config.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(config.get('default_scope', 'mmdet'))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is None:
        warnings.simplefilter('once')
        warnings.warn('checkpoint is None, use COCO classes by default.')
        model.dataset_meta = {'classes': get_classes('coco')}
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get('meta', {})

        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            # mmdet 3.x, all keys should be lowercase
            model.dataset_meta = {
                k.lower(): v
                for k, v in checkpoint_meta['dataset_meta'].items()
            }
        elif 'CLASSES' in checkpoint_meta:
            # < mmdet 3.x
            classes = checkpoint_meta['CLASSES']
            model.dataset_meta = {'classes': classes}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}

    # Priority:  args.palette -> config -> checkpoint
    if palette != 'none':
        model.dataset_meta['palette'] = palette
    else:
        test_dataset_cfg = copy.deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def inference_detector(
    model: nn.Module,
    imgs: ImagesType,
    test_pipeline: Optional[Compose] = None,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    result_list = []
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)

        if text_prompt:
            data_['text'] = text_prompt
            data_['custom_entities'] = custom_entities

        # build the data pipeline
        data_ = test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            results = model.test_step(data_)[0]

        result_list.append(results)

    if not is_batch:
        return result_list[0]
    else:
        return result_list


# TODO: Awaiting refactoring
async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromNDArray'

    # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    for m in model.modules():
        assert not isinstance(
            m,
            RoIPool), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(data, rescale=True)
    return results


def build_test_pipeline(cfg: ConfigType) -> ConfigType:
    """Build test_pipeline for mot/vis demo. In mot/vis infer, original
    test_pipeline should remove the "LoadImageFromFile" and
    "LoadTrackAnnotations".

    Args:
         cfg (ConfigDict): The loaded config.
    Returns:
         ConfigType: new test_pipeline
    """
    # remove the "LoadImageFromFile" and "LoadTrackAnnotations" in pipeline
    transform_broadcaster = cfg.test_dataloader.dataset.pipeline[0].copy()
    for transform in transform_broadcaster['transforms']:
        if transform['type'] == 'Resize':
            transform_broadcaster['transforms'] = transform
    pack_track_inputs = cfg.test_dataloader.dataset.pipeline[-1].copy()
    test_pipeline = Compose([transform_broadcaster, pack_track_inputs])

    return test_pipeline


def inference_mot(model: nn.Module, img: np.ndarray, frame_id: int,
                  video_len: int) -> SampleList:
    """Inference image(s) with the mot model.

    Args:
        model (nn.Module): The loaded mot model.
        img (np.ndarray): Loaded image.
        frame_id (int): frame id.
        video_len (int): demo video length
    Returns:
        SampleList: The tracking data samples.
    """
    cfg = model.cfg
    data = dict(
        img=[img.astype(np.float32)],
        frame_id=[frame_id],
        ori_shape=[img.shape[:2]],
        img_id=[frame_id + 1],
        ori_video_length=[video_len])

    test_pipeline = build_test_pipeline(cfg)
    data = test_pipeline(data)

    if not next(model.parameters()).is_cuda:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        data = default_collate([data])
        result = model.test_step(data)[0]
    return result


def init_track_model(config: Union[str, Config],
                     checkpoint: Optional[str] = None,
                     detector: Optional[str] = None,
                     reid: Optional[str] = None,
                     device: str = 'cuda:0',
                     cfg_options: Optional[dict] = None) -> nn.Module:
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (Optional[str], optional): Checkpoint path. Defaults to
            None.
        detector (Optional[str], optional): Detector Checkpoint path, use in
            some tracking algorithms like sort.  Defaults to None.
        reid (Optional[str], optional): Reid checkpoint path. use in
            some tracking algorithms like sort. Defaults to None.
        device (str, optional): The device that the model inferences on.
            Defaults to `cuda:0`.
        cfg_options (Optional[dict], optional): Options to override some
            settings in the used config. Defaults to None.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get('meta', {})
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            if 'CLASSES' in checkpoint_meta['dataset_meta']:
                value = checkpoint_meta['dataset_meta'].pop('CLASSES')
                checkpoint_meta['dataset_meta']['classes'] = value
            model.dataset_meta = checkpoint_meta['dataset_meta']

    if detector is not None:
        assert not (checkpoint and detector), \
            'Error: checkpoint and detector checkpoint cannot both exist'
        load_checkpoint(model.detector, detector, map_location='cpu')

    if reid is not None:
        assert not (checkpoint and reid), \
            'Error: checkpoint and reid checkpoint cannot both exist'
        load_checkpoint(model.reid, reid, map_location='cpu')

    # Some methods don't load checkpoints or checkpoints don't contain
    # 'dataset_meta'
    # VIS need dataset_meta, MOT don't need dataset_meta
    if not hasattr(model, 'dataset_meta'):
        warnings.warn('dataset_meta or class names are missed, '
                      'use None by default.')
        model.dataset_meta = {'classes': None}

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model
