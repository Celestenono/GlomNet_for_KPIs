from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from openslide import OpenSlide
import os
import matplotlib.pyplot as plt
import numpy as np
from source.utilities.rgb2gray import rgb2gray
import random
from source.dataloading.cross_validation_split import cross_validation_split
import json

MAGNIFICATIONS = {"56Nx": 80, "DN": 80, "NEP25": 40, "normal": 80}
# PATCH_SIZE = {"56Nx": 2048, "DN": 2048, "NEP25": 1024, "normal": 2048}
PATCH_SIZE = {"56Nx": 1024, "DN": 1024, "NEP25": 512, "normal": 1024}

def reorga_and_shuffle_data(json_patches):
    list_glom = json_patches["glomerulus"].copy()
    list_non_glom = json_patches["non_glomerulus"].copy()
    list_index_non_glom = list(range(0, len(list_non_glom)))
    for patch_dict_i in range(0, len(list_glom)):
        patch_dict = list_glom[patch_dict_i]
        if patch_dict == {}:
            if len(list_index_non_glom) == 0:
                list_index_non_glom = list(range(0, len(list_non_glom)))
            patch_id = list_index_non_glom.pop(random.randrange(len(list_index_non_glom)))
            patch_dict = list_non_glom[patch_id]
            list_glom[patch_dict_i] = patch_dict
    random.shuffle(list_glom)
    json_patches["all_shuffle"] = list_glom
    return json_patches

def clean_dataset(json_patches, path_data):
    list_data = json_patches["all_shuffle"]
    new_list = []
    nb_glom = 0
    for patch_dict_i in range(0, len(list_data)):
        patch_dict = list_data[patch_dict_i]
        image_filename = patch_dict["image_name"] + "wsi.tiff"
        mask_filename = patch_dict["image_name"] + "mask.tiff"
        dir = patch_dict["dir"]
        magnification = patch_dict["magnification"]
        patch_size_patch = patch_dict["patch_size"]
        x = patch_dict["x"]
        y = patch_dict["y"]
        image_openslide = OpenSlide(path_data + "/" + dir + "/" + image_filename)
        image_size = image_openslide.dimensions
        if x+patch_size_patch[0] < image_size[0] and y+patch_size_patch[1] < image_size[1]:
            new_list.append(patch_dict)
            if patch_dict["glom"]:
                nb_glom += 1
        else:
            print(image_filename, x, y)
        # else:
        #     print("to small")
    print(len(list_data))
    print(len(new_list))
    print(nb_glom)
    json_patches["all_shuffle"] = new_list
    return json_patches

def patch_extraction(path_data, data_list = []):
    dirs = ["56Nx", "DN", "NEP25", "normal"]
    json_patches = {}
    json_patches["non_glomerulus"] = []
    json_patches["glomerulus"] = []
    for dir in dirs:
        patch_size = (PATCH_SIZE[dir], PATCH_SIZE[dir])
        if data_list == [] :
            data_list = os.listdir(path_data + "/" + dir + "/")
        for image_filename in data_list:
            if os.path.exists(path_data + "/" + dir + "/" + image_filename):
                print(image_filename)
                image_openslide = OpenSlide(path_data + "/" + dir + "/" + image_filename)
                image_size = image_openslide.dimensions
                print(image_size)

                mask_filename = image_filename[:-8] + "mask.tiff"
                mask_openslide = OpenSlide(path_data + "/" + dir + "/" + mask_filename)
                mask_size = mask_openslide.dimensions
                print(mask_size)

                x_possibility = []
                y_possibility = []
                for x in range(0, image_size[0], patch_size[0] // 4):
                    x_possibility.append(x)
                    for y in range(0, image_size[1], patch_size[1] // 4):
                        y_possibility.append(y)
                        patch_image = np.array(image_openslide.read_region((x, y), 0, patch_size))
                        patch_image_gray = rgb2gray(patch_image)
                        patch_mask = np.array(mask_openslide.read_region((x, y), 0, patch_size))[:, :, 0]
                        patch_image_gray[patch_image_gray > 190] = 0 #210, 190
                        if np.count_nonzero(patch_image_gray) > (patch_size[0] * patch_size[1] / 4) and \
                                x+patch_size[0] < image_size[0] and y+patch_size[1] < image_size[1]:
                            patch_info = {}
                            patch_info["image_name"] = image_filename[:-8]
                            patch_info["dir"] = dir
                            patch_info["x"] = x
                            patch_info["y"] = y
                            patch_info["magnification"] = MAGNIFICATIONS[dir]
                            patch_info["patch_size"] = patch_size
                            if np.count_nonzero(patch_mask) > (patch_size[0]*patch_size[1]/20):
                                patch_info["glom"] = True
                                json_patches["glomerulus"].append(patch_info.copy())
                                json_patches["glomerulus"].append({})
                                json_patches["glomerulus"].append({})
                            elif np.count_nonzero(patch_mask) <= (patch_size[0]*patch_size[1]/20) and np.count_nonzero(
                                    patch_mask) > 0:
                                pass
                            else:
                                patch_info["glom"] = False
                                json_patches["non_glomerulus"].append(patch_info.copy())
    print(len(json_patches["glomerulus"]))
    print(len(json_patches["non_glomerulus"]))
    json_patches = reorga_and_shuffle_data(json_patches)
    return json_patches

if __name__ == "__main__":
    raw_training_directory = "/Users/nmoreau/Documents/KPIs_challenge/KPIs24 Training Data/Task2_WSI_level/"
    cross_validation_fold = cross_validation_split(fold_num=0)
    data_train = patch_extraction(raw_training_directory, data_list=cross_validation_fold["train"])
    data_val = patch_extraction(raw_training_directory, data_list=cross_validation_fold["val"])
    # with open(raw_training_directory + "/data_train_bis.json", "w") as json_file:
    #     json_file.write(json.dumps(data_train))
    # with open(raw_training_directory + "/data_val_bis.json", "w") as json_file:
    #     json_file.write(json.dumps(data_val))

    # raw_training_directory = "/Users/nmoreau/Documents/KPIs_challenge/KPIs24 Training Data/Task2_WSI_level/"
    # with open(raw_training_directory + "/data_train.json", "r") as json_file:
    #     data_train = json.load(json_file)
    # with open(raw_training_directory + "/data_val.json", "r") as json_file:
    #     data_val = json.load(json_file)
    # data_train = clean_dataset(data_train, raw_training_directory)
    # data_val = clean_dataset(data_val, raw_training_directory)
    # with open(raw_training_directory + "/data_train_correct.json", "w") as json_file:
    #     json_file.write(json.dumps(data_train))
    # with open(raw_training_directory + "/data_val_correct.json", "w") as json_file:
    #     json_file.write(json.dumps(data_val))
    # dir = "DN"
    # patch_size = (PATCH_SIZE[dir], PATCH_SIZE[dir])
    # json_patches = {}
    # json_patches["infos"] = {"patch_size": patch_size}
    # json_patches["non_glomerulus"] = []
    # json_patches["glomerulus"] = []
    # raw_training_directory = "/Users/nmoreau/Documents/KPIs_challenge/KPIs24 Training Data/Task2_WSI_level/" + dir + "/"
    # for filename in os.listdir(raw_training_directory):
    #     if filename.endswith("wsi.tiff"):
    #         print(filename)
    #         image_openslide = OpenSlide(raw_training_directory + filename)
    #         image_size = image_openslide.dimensions
    #         print(image_size)
    #
    #         maskname = filename[:-8] + "mask.tiff"
    #         mask_openslide = OpenSlide(raw_training_directory + maskname)
    #         mask_size = mask_openslide.dimensions
    #         print(mask_size)
    #
    #         x_possibility = []
    #         y_possibility = []
    #         i = 0
    #         for x in range(0, image_size[0], patch_size[0] // 4):
    #             x_possibility.append(x)
    #             for y in range(0, image_size[1], patch_size[1] // 4):
    #                 y_possibility.append(y)
    #                 patch_image = np.array(image_openslide.read_region((x, y), 0, patch_size))
    #                 patch_image_gray = rgb2gray(patch_image)
    #                 patch_mask = np.array(mask_openslide.read_region((x, y), 0, patch_size))[:, :, 0]
    #                 patch_image_gray[patch_image_gray > 190] = 0 #210, 190
    #                 if np.count_nonzero(patch_image_gray) > (patch_size[0] * patch_size[1] / 4):
    #                     patch_info = {}
    #                     patch_info["image_name"] = filename[:-8]
    #                     patch_info["dir"] = dir
    #                     patch_info["x"] = x
    #                     patch_info["y"] = y
    #                     patch_info["magnification"] = MAGNIFICATIONS[dir]
    #                     patch_info["patch_size"] = PATCH_SIZE[dir]
    #                     if np.count_nonzero(patch_mask) > (patch_size[0]*patch_size[1]/20):
    #                         patch_info["glom"] = True
    #                         json_patches["glomerulus"].append(patch_info.copy())
    #                         json_patches["glomerulus"].append({})
    #                         json_patches["glomerulus"].append({})
    #                     elif np.count_nonzero(patch_mask) <= (patch_size[0]*patch_size[1]/20) and np.count_nonzero(
    #                             patch_mask) > 0:
    #                         pass
    #                     else:
    #                         patch_info["glom"] = False
    #                         json_patches["non_glomerulus"].append(patch_info.copy())
    # print(len(json_patches["glomerulus"]))
    # print(len(json_patches["non_glomerulus"]))
    # json_patches = reorga_and_shuffle_data(json_patches)