import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from fastai.layers import FlattenedLoss

USE_GPU = torch.cuda.is_available()


class GeneralizedDiceLoss(_Loss):
    # reference: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/loss_segmentation.html#generalised_dice_loss
    def __init__(self, **kwargs):
        super(GeneralizedDiceLoss, self).__init__(**kwargs)
        self.softmax = nn.Softmax(1)

    def forward(self, input, target):
        prediction = self.softmax(input)

        if USE_GPU:
            one_hot = torch.sparse.torch.eye(2).cuda().index_select(0, target.long())
        else:
            one_hot = torch.sparse.torch.eye(2).index_select(0, target.long())

        ref_vol = torch.sum(one_hot, 0)

        seg_vol = torch.sum(prediction, 0)
        intersect = torch.sum(one_hot * prediction, 0)

        weights = torch.reciprocal(ref_vol ** 2)
        weights[weights == float("Inf")] = 0

        generalized_dice_numerator = 2 * torch.sum(weights * intersect)
        generalized_dice_denominator = torch.sum(
            weights * torch.max(seg_vol + ref_vol, torch.ones_like(weights))
        )
        generalized_dice_score = (
            generalized_dice_numerator / generalized_dice_denominator
        )

        generalized_dice_score[torch.isnan(generalized_dice_score)] = 1.0
        return 1 - generalized_dice_score


generalized_dice_loss = FlattenedLoss(GeneralizedDiceLoss, axis=1)
