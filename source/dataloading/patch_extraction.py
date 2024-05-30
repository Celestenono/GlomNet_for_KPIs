import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from source.utilities.rgb2gray import rgb2gray
import random
def patch_extraction(path_data, patch_size = (256,256), data_list = []):
    path_images = path_data + "/images/"
    path_masks = path_data + "/masks/"
    json_patches = {}
    json_patches["infos"] = {"patch_size": patch_size}
    json_patches["non_glomerulus"] = []
    json_patches["glomerulus"] = []
    if data_list == [] :
        data_list = os.listdir(path_masks)
    for mask in data_list:
        if os.path.exists(path_masks + mask):
            image_mask = Image.open(path_masks + mask)
            array_mask = np.array(image_mask)
            array_mask[array_mask > 0] = 1
            if os.path.exists(path_images + mask):
                print(mask)
                image = Image.open(path_images+ mask)
                array = np.array(image)
                array_gray = rgb2gray(array)
                array_size = array_gray.shape
                x_possibility = []
                y_possibility = []
                for x in range(0, array_size[0], patch_size[0]//4):
                    x_possibility.append(x)
                    for y in range(0, array_size[1], patch_size[1]//4):
                        y_possibility.append(y)
                        patch_image = array_gray[x:x+patch_size[0], y:y+patch_size[1]]
                        patch_mask = array_mask[x:x+patch_size[0], y:y+patch_size[1]]
                        if np.count_nonzero(patch_image) > (patch_size[0]*patch_size[1]/4) and patch_mask.shape == patch_size:
                            patch_info = {}
                            patch_info["image_name"] = mask
                            patch_info["x"] = x
                            patch_info["y"] = y
                            if np.count_nonzero(patch_mask) > (patch_size[0]*patch_size[1]/10) :
                                patch_info["glom"] = True
                                json_patches["glomerulus"].append(patch_info.copy())
                                json_patches["glomerulus"].append({})
                                json_patches["glomerulus"].append({})
                            elif np.count_nonzero(patch_mask) <= (patch_size[0]*patch_size[1]/10) and np.count_nonzero(patch_mask) > 0 :
                                pass
                            else :
                                patch_info["glom"] = False
                                json_patches["non_glomerulus"].append(patch_info.copy())
    json_patches = reorga_and_shuffle_data(json_patches)
    return json_patches
def reorga_and_shuffle_data(json_patches):
    list_glom = json_patches["glomerulus"]
    list_non_glom = json_patches["non_glomerulus"]
    list_index_non_glom = list(range(0, len(list_non_glom)))
    for patch_dict_i in range(0, len(list_glom)):
        patch_dict = list_glom[patch_dict_i]
        if patch_dict == {}:
            patch_id = list_index_non_glom.pop(random.randrange(len(list_index_non_glom)))
            patch_dict = list_non_glom[patch_id]
            list_glom[patch_dict_i] = patch_dict
    random.shuffle(list_glom)
    json_patches["all_shuffle"] = list_glom
    return json_patches