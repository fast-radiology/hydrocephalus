from fastai.vision import unet_learner, models

from losses import generalized_dice_loss


def get_learner(data, metrics=None):
    if metrics is None:
        metrics = []

    return unet_learner(
        data,
        models.resnet34,
        metrics=metrics,
        self_attention=True,
        loss_func=generalized_dice_loss,
        wd=1e-7,
    ).to_fp16()
