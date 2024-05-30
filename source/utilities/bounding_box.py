import numpy as np


def bbox(array: np.ndarray) -> np.ndarray:
    '''
    Compute the smallest bounding box around the mask
    :param array: 2D array usually mask
    :return: bounding box (Xmin, Xmax, Ymin, Ymax)
    '''
    a = np.where(array != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox
