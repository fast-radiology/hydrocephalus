from fastai.vision import unet_learner, models

from hydrocephalus.losses import generalized_dice_loss, USE_GPU


def get_learner(data, metrics=None, model_dir='models'):
    if metrics is None:
        metrics = []

    learner = unet_learner(
        data,
        models.resnet34,
        metrics=metrics,
        self_attention=True,
        loss_func=generalized_dice_loss,
        wd=1e-7,
        model_dir=model_dir,
    )
    if USE_GPU:
        return learner.to_fp16()
    return learner
