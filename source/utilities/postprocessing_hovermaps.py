import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from source.utilities.remove_small_objects import remove_small_objects

def postprocessing_hovermaps(pred):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred_bin = pred["inf_results_bin"]
    pred_ho = pred["inf_results_ho"]
    pred_ver = pred["inf_results_ver"]


    pred_bin = ndi.label(pred_bin)[0]
    pred_bin = remove_small_objects(pred_bin, min_size=10)
    pred_bin[pred_bin > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        pred_ho, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        pred_ver, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    borders = np.maximum(sobelh, sobelv)
    borders[borders < 0.7] = 0
    borders[borders >= 0.7] = 1
    borders = ndi.binary_dilation(borders, structure=np.ones((20, 20))).astype(int)

    distance = ndi.distance_transform_edt(pred_bin - borders)
    markers = distance.copy()
    markers[distance < 50] = 0
    markers[distance >= 50] = 1
    markers, _ = ndi.label(markers)
    pred_inst = watershed(-distance, markers, mask=pred_bin)
    return pred_inst