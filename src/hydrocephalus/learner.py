from fastai.vision import unet_learner, models

from hydrocephalus.losses import generalized_dice_loss


def get_learner(data, metrics=None, model_dir='models', engine='GPU'):
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
    if engine == "GPU":
        return learner.to_fp16()
    return learner
