import os
from pathlib import Path
import pandas as pd
from fastai.vision import get_transforms, imagenet_stats, SegmentationItemList


CODES = ['void', 'water']


def get_scans(data_path):
    scans = []
    for root, dirs, files in os.walk(data_path):
        if 'CT' in root:
            for _file in files:
                if _file.endswith('.dcm'):
                    scans.append(root + '/' + _file)

    return sorted(scans)


def get_y_fn(path):
    return str('.' / Path(path).parent / '../label' / Path(path).name)


def get_data(scans, valid_func, bs, size):
    return (
        SegmentationItemList.from_df(pd.DataFrame(scans, columns=['files']), '.')
        .split_by_valid_func(valid_func)
        .label_from_func(get_y_fn, classes=CODES)
        .transform(
            get_transforms(max_rotate=5.0, max_lighting=0, p_lighting=0, max_warp=0),
            size=size,
            tfm_y=True,
        )
        .databunch(bs=bs, num_workers=0)  # one worker for reproducibility
        .normalize(imagenet_stats)
    )
