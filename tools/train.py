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
        return packed_results


@TRANSFORMS.register_module()
class DebugInput:
    def __call__(self, results):
        img = results['img']
        # print(f"Final input - Shape: {img.shape}", flush=True)
        # print(f"Channel means: {img.mean(dim=[1,2])}", flush=True)
        # print(f"Keys in results: {list(results.keys())}", flush=True)
        # print(f"Meta info preview: {[k for k in ['img_id', 'img_shape', 'ori_shape', 'scale_factor'] if k in results]}", flush=True)
        return results

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()

torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
