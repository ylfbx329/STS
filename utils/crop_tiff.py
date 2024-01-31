import os
import random
import shutil

import numpy as np
from osgeo import gdal

from gdal_save import save_tif

os.environ['GDAL_DATA'] = r'D:\Anaconda3\envs\rs\Library\share\gdal'
os.environ['PROJ_LIB'] = r"D:\Anaconda3\envs\rs\Library\share\proj"

random.seed(1)

gdal.AllRegister()

# MODIS-哨兵2
# s2_16_01 = gdal.Open('../data/s2_16_01.tif')
# s2_17_01 = gdal.Open('../data/s2_17_01.tif')
# mod_01 = gdal.Open('../data/mod_01.tif')
#
# s2_16_02 = gdal.Open('../data/s2_16_02.tif')
# s2_17_02 = gdal.Open('../data/s2_17_02.tif')
# mod_02 = gdal.Open('../data/mod_02.tif')

# CIA 10月08-10月17 landsat-modis
mod_08 = gdal.Open(r'D:\useful\program\rs\STS\data\M2001_281_08oct.tif')
lan_08 = gdal.Open(r'D:\useful\program\rs\STS\data\L2001_281_08oct.tif')
lan_17 = gdal.Open(r'D:\useful\program\rs\STS\data\L2001_290_17oct.tif')


# MODIS-哨兵2
# def save_random_patches(s2_16_01, s2_17_01, mod_01, s2_16_02, s2_17_02, mod_02, save_path, patch_size=(256, 256), train_size=500):
#     """
#     TODO 规避no data
#     输入img的shape为h,w,c
#     :param s2_train_img:
#     :param s2_label_img:
#     :param save_path:
#     :param patch_size:
#     :param train_size:
#     :return:
#     """
#     s2_16_01_arr = s2_16_01.ReadAsArray()  # c,h,w
#     s2_17_01_arr = s2_17_01.ReadAsArray()
#     mod_01_arr = mod_01.ReadAsArray()
#
#     # mod_01_arr_mean = mod_01_arr.mean(axis=0)
#     # print(mod_01_arr_mean.shape)
#
#     img_01 = np.concatenate((mod_01_arr, s2_16_01_arr, s2_17_01_arr), axis=0)  # 7+4+4,h,w
#     # print(img_01.shape)
#
#     del s2_16_01_arr, s2_17_01_arr, mod_01_arr
#
#     s2_16_02_arr = s2_16_02.ReadAsArray()
#     s2_17_02_arr = s2_17_02.ReadAsArray()
#     mod_02_arr = mod_02.ReadAsArray()
#
#     # mod_02_arr_mean = mod_02_arr.mean(axis=0)
#     # print(mod_02_arr_mean.shape)
#
#     img_02 = np.concatenate((mod_02_arr, s2_16_02_arr, s2_17_02_arr), axis=0)
#
#     del s2_16_02_arr, s2_17_02_arr, mod_02_arr
#
#     for i in range(0, train_size):
#         img_idx = random.choice([1, 2])
#         img = img_01 if img_idx == 1 else img_02
#         # if img_idx == 1:
#         upper_left_x = random.randrange(0, img.shape[2] - patch_size[1])
#         upper_left_y = random.randrange(0, img.shape[1] - patch_size[0])
#         crop_point = [upper_left_x,
#                       upper_left_y,
#                       upper_left_x + patch_size[0],
#                       upper_left_y + patch_size[1]]
#         print(i, crop_point)
#         patch = img[:, crop_point[1]:crop_point[3], crop_point[0]:crop_point[2]]
#         os.makedirs(save_path + "/trainA", exist_ok=True)
#         os.makedirs(save_path + "/trainB", exist_ok=True)
#         if img_idx == 1:
#             save_tif(patch[:11, :, :], upper_left_x, upper_left_y, s2_16_01, save_path + "/trainA/" + str(i) + ".tif")
#             save_tif(patch[11:, :, :], upper_left_x, upper_left_y, s2_17_01, save_path + "/trainB/" + str(i) + ".tif")
#         else:
#             save_tif(patch[:11, :, :], upper_left_x, upper_left_y, s2_16_02, save_path + "/trainA/" + str(i) + ".tif")
#             save_tif(patch[11:, :, :], upper_left_x, upper_left_y, s2_17_02, save_path + "/trainB/" + str(i) + ".tif")
#
#     print('Done!')


# save_random_patches(s2_16_01, s2_17_01, mod_01, s2_16_02, s2_17_02, mod_02, save_path="../data")


def crop_tiff(mod_08, lan_08, lan_17, save_path, patch_size=256, train_size=0):
    mod_08_arr = (mod_08.ReadAsArray() / 10000).astype(np.float32)  # c,h,w
    lan_08_arr = (lan_08.ReadAsArray() / 10000).astype(np.float32)
    lan_17_arr = (lan_17.ReadAsArray() / 10000).astype(np.float32)

    img = np.concatenate((mod_08_arr, lan_08_arr, lan_17_arr,), axis=0)  # 6+6+6,h,w
    print(img.shape)

    del mod_08_arr, lan_08_arr, lan_17_arr

    try:
        shutil.rmtree(save_path + "/trainA")
        shutil.rmtree(save_path + "/trainB")
        shutil.rmtree(save_path + "/testA")
        shutil.rmtree(save_path + "/testB")
    except:
        pass
    finally:
        os.makedirs(save_path + "/trainA", exist_ok=True)
        os.makedirs(save_path + "/trainB", exist_ok=True)
        os.makedirs(save_path + "/testA", exist_ok=True)
        os.makedirs(save_path + "/testB", exist_ok=True)

    for i in range(0, train_size):
        upper_left_x = random.randrange(0, img.shape[2] - patch_size)
        upper_left_y = random.randrange(0, img.shape[1] - patch_size)
        crop_point = [upper_left_x,
                      upper_left_y,
                      upper_left_x + patch_size,
                      upper_left_y + patch_size]
        # print(i, crop_point)
        patch = img[:, crop_point[1]:crop_point[3], crop_point[0]:crop_point[2]]
        save_tif(patch[:12, :, :], upper_left_x, upper_left_y, gdal.GDT_Float32, lan_08, save_path + "/trainA/" + str(i) + ".tif")
        save_tif(patch[12:, :, :], upper_left_x, upper_left_y, gdal.GDT_Float32, lan_08, save_path + "/trainB/" + str(i) + ".tif")

    test_size = 0
    for upper_left_x in range(0, img.shape[2] - patch_size + 1, patch_size):
        for upper_left_y in range(0, img.shape[1] - patch_size + 1, patch_size):
            crop_point = [upper_left_x,
                          upper_left_y,
                          upper_left_x + patch_size,
                          upper_left_y + patch_size]
            patch = img[:, crop_point[1]:crop_point[3], crop_point[0]:crop_point[2]]
            save_tif(patch[:12, :, :], upper_left_x, upper_left_y, gdal.GDT_Float32, lan_08, save_path + "/testA/" + str(test_size) + ".tif")
            save_tif(patch[12:, :, :], upper_left_x, upper_left_y, gdal.GDT_Float32, lan_08, save_path + "/testB/" + str(test_size) + ".tif")
            test_size += 1
    if img.shape[1] % patch_size != 0:
        for upper_left_x in range(0, img.shape[2] - patch_size + 1, patch_size):
            upper_left_y = img.shape[1] - patch_size
            crop_point = [upper_left_x,
                          upper_left_y,
                          upper_left_x + patch_size,
                          upper_left_y + patch_size]
            patch = img[:, crop_point[1]:crop_point[3], crop_point[0]:crop_point[2]]
            save_tif(patch[:12, :, :], upper_left_x, upper_left_y, gdal.GDT_Float32, lan_08, save_path + "/testA/" + str(test_size) + ".tif")
            save_tif(patch[12:, :, :], upper_left_x, upper_left_y, gdal.GDT_Float32, lan_08, save_path + "/testB/" + str(test_size) + ".tif")
            test_size += 1
    if img.shape[2] % patch_size != 0:
        for upper_left_y in range(0, img.shape[1] - patch_size + 1, patch_size):
            upper_left_x = img.shape[2] - patch_size
            crop_point = [upper_left_x,
                          upper_left_y,
                          upper_left_x + patch_size,
                          upper_left_y + patch_size]
            patch = img[:, crop_point[1]:crop_point[3], crop_point[0]:crop_point[2]]
            save_tif(patch[:12, :, :], upper_left_x, upper_left_y, gdal.GDT_Float32, lan_08, save_path + "/testA/" + str(test_size) + ".tif")
            save_tif(patch[12:, :, :], upper_left_x, upper_left_y, gdal.GDT_Float32, lan_08, save_path + "/testB/" + str(test_size) + ".tif")
            test_size += 1

    print(test_size)
    print('Done!')


crop_tiff(mod_08, lan_08, lan_17, save_path="../data")
