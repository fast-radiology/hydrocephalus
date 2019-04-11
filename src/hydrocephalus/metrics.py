import numpy as np
import pydicom


def dice(preds, targs, iou: bool = False):
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum().float()
    union = (preds + targs).sum().float()

    if not iou:
        return 2.0 * intersect / union
    else:
        return intersect / (union - intersect)


def iou(preds, target):
    return dice(preds, target, iou=True)


def accuracy(preds, target):
    """
    The one defined in fastai assumes different input format
    """
    target = target.cpu()
    return (preds.cpu() == target).float().mean()


def tp(preds, target):
    return ((preds == True) & (target == True)).sum()


def fp(preds, target):
    return ((preds == True) & (target == False)).sum()


def fn(preds, target):
    return ((preds == False) & (target == True)).sum()


def get_volume(image, spacing):
    return image.sum().item() * spacing.prod()


def get_examination_spacing(scans):
    scan = pydicom.dcmread(scans[0])
    next_scan = pydicom.dcmread(scans[1])
    thickness = float(next_scan.ImagePositionPatient[-1]) - float(
        scan.ImagePositionPatient[-1]
    )
    spacing = np.array([thickness] + list(scan.PixelSpacing), dtype=np.float32)
    return spacing


def get_result(preds, true, scans, metrics, split, examination):
    result = {'split': split, 'examination': examination}
    for metric in metrics:
        result[metric.__name__] = metric(preds, true).item()

    spacing = get_examination_spacing(scans)
    result['preds_volume'] = get_volume(preds, spacing)
    result['true_volume'] = get_volume(true, spacing)

    if {'tp', 'fp', 'fn'}.issubset(set(map(lambda f: f.__name__, metrics))):
        tp = result['tp']
        fp = result['fp']
        fn = result['fn']

        result['precision'] = tp / float(tp + fp)
        result['recall'] = tp / float(tp + fn)
        result['volumetric_similarity'] = 1 - abs(fn - fp) / float(2 * tp + fp + fn)
    return result
