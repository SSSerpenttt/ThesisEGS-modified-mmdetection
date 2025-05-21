import torch
from mmdet.models.dense_heads.reppoints_head import RepPointsHead
from mmdet.registry import MODELS
from mmengine.model import multi_apply

@MODELS.register_module()
class RepPointsRPNHead(RepPointsHead):
    def __init__(self, **kwargs):
        kwargs['num_classes'] = 1  # RPN is binary: object vs. background
        super().__init__(**kwargs)

    def get_proposals(self, cls_scores, bbox_preds, score_factors, img_metas, cfg):
        proposal_list = self.get_bboxes(
            cls_scores, bbox_preds, score_factors, img_metas, cfg)
        return proposal_list

    def loss_and_proposals(self, x, batch_data_samples, proposal_cfg):
        losses = self.loss(x, batch_data_samples)
        if proposal_cfg is not None:
            cls_scores, bbox_preds, score_factors = self.predict_raw(x)
            proposal_list = self.get_proposals(
                cls_scores, bbox_preds, score_factors,
                [sample.metainfo for sample in batch_data_samples],
                proposal_cfg
            )
        else:
            proposal_list = None
        return losses, proposal_list

    def simple_test_rpn(self, x, img_metas):
        cls_scores, bbox_preds, score_factors = self.predict_raw(x)
        proposal_list = self.get_proposals(
            cls_scores, bbox_preds, score_factors, img_metas, self.test_cfg)
        return proposal_list

    def predict_raw(self, x):
        return multi_apply(self.forward_single, x)
