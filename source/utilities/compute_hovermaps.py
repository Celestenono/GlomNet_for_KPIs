import numpy as np
import skimage


def compute_hovermaps(instance_mask: np.ndarray):
    h_map = instance_mask.astype("float32", copy=True)
    v_map = instance_mask.astype("float32", copy=True)
    for region in skimage.measure.regionprops(instance_mask):
        v_dist = region.coords[:, 0] - region.centroid[0]
        h_dist = region.coords[:, 1] - region.centroid[1]

        h_dist[h_dist < 0] /= -np.amin(h_dist)
        h_dist[h_dist > 0] /= np.amax(h_dist)

        v_dist[v_dist < 0] /= -np.amin(v_dist)
        v_dist[v_dist > 0] /= np.amax(v_dist)

        h_map[h_map == region.label] = h_dist
        v_map[v_map == region.label] = v_dist
    hv_maps = np.array([h_map, v_map])
    return hv_maps
