from pathlib import Path
import pandas as pd
from fastai.vision import get_transforms, imagenet_stats, unet_learner, models

from losses import generalized_dice_loss


CODES = ['void', 'water']


def get_y_fn(path):
    return str('.' / Path(path).parent / '../label' / Path(path).name)


def get_data(scans, valid_func, bs, size):
    return (
        SegmentationItemList.from_df(pd.DataFrame(scans, columns=['files']), '.')
        .split_by_valid_func(val_examination_filtering_func)
        .label_from_func(get_y_fn, classes=CODES)
        .transform(
            get_transforms(max_rotate=5.0, max_lighting=0, p_lighting=0, max_warp=0),
            size=size,
            tfm_y=True,
        )
        .databunch(bs=bs, num_workers=0)  # one worker for reproducibility
        .normalize(imagenet_stats)
    )


def get_learner(data):
    return unet_learner(
        data,
        models.resnet34,
        metrics=[],
        self_attention=True,
        loss_func=generalized_dice_loss,
        wd=1e-7,
    ).to_fp16()
