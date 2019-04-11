import pydicom
import cv2
import numpy as np

from fastai.vision import Image, ImageSegment, pil2tensor


def open_dcm_image(fn, *args, **kwargs) -> Image:
    window_min = -100
    window_max = 100

    array = pydicom.dcmread(fn).pixel_array
    array = np.clip(array, a_min=window_min, a_max=window_max)

    array = ((array - window_min) / (window_max - window_min) * (255 - 0) + 0).astype(
        np.uint8
    )
    array = cv2.equalizeHist(array.astype(np.uint8))

    array = np.repeat(array[:, :, None], 3, axis=2)

    # we can store images in this format :top: to make stuff faster...
    return Image(pil2tensor(array, np.float32).div_(255))


def open_dcm_mask(fn, *args, **kwargs) -> Image:
    x = pydicom.dcmread(fn).pixel_array
    x = pil2tensor(x, np.float32)
    return ImageSegment(x)


def get_shape(filename):
    return np.array(pydicom.dcmread(filename).pixel_array.shape)
