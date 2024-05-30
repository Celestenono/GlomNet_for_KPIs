import numpy as np


def rgb2gray(rgb_array: np.ndarray) -> np.ndarray:
    '''
    Convert a rgb array to a greyscale array
    :param rgb_array: array with 3 or 4 channels
    :return: 2D array (no channel)
    '''
    r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.astype(np.uint8)

    return gray
