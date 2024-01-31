from pprint import pprint

import imgvision as iv
import torch
from osgeo import gdal
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from einops import rearrange
from utils.gdal_save import save_tif


def evaluate(arr1, arr2):
    """
    SAM较小的值通常表示更好的相似性或匹配
    MSE均方误差值越小，表示两个图片越相似
    PSNR的值通常以分贝（dB）为单位，越高的PSNR值表示图像质量越好
    SSIM值在-1到1之间，越接近1表示图像越相似
    ERGAS，越小的值代表更好的融合效果
    """
    Metric = iv.spectra_metric(arr1, arr2, scale=1)

    # 评价MSE：
    MSE = Metric.MSE()
    # 评价SAM：
    SAM = Metric.SAM()
    # 评价PSNR：    np.mean(10 * np.log10(np.power(np.max(self.max_v, axis=0), 2) / self.MSE('mat')))
    PSNR = Metric.PSNR()
    # 评价SSIM：
    SSIM = Metric.SSIM()
    # 评价ERGAS:
    ERGAS = Metric.ERGAS()
    # 评价PSNR, SAM, ERGAS, SSIM
    # Metric.Evaluation()
    # print("MSE:", MSE, "SAM:", SAM, "PSNR:", PSNR, "SSIM:", SSIM, "ERGAS:", ERGAS)
    return MSE, SAM, PSNR, SSIM, ERGAS


def min_max_normalize(arr):
    # 获取每个通道的最大值和最小值
    min_values = np.min(arr, axis=(0, 1))
    max_values = np.max(arr, axis=(0, 1))

    print(min_values, max_values)

    # 对每个通道进行最大最小归一化
    normalized_arr = (arr - min_values) / (max_values - min_values)

    return normalized_arr


if __name__ == '__main__':
    # real_net = gdal.Open(r"D:\useful\program\rs\CycleGAN\results\sts_cyclegan\test_latest\images\0_real_B.tif")
    real_path = r"D:\useful\program\rs\CycleGAN\results\cia_cyclegan\test_latest\images\\"
    # up_path = r"D:\useful\program\rs\CycleGAN\results\cia_cyclegan\test_latest\images\\"
    res_path = r"D:\useful\program\rs\CycleGAN\results\cia_cyclegan\test_latest\images\\"

    # up_metric = []
    res_metric = []
    rec_metric = []

    for i in range(500):
        real_dataset = gdal.Open(real_path + str(i) + "_real_B.tif")
        # up_dataset = gdal.Open(up_path + str(i) + "_real_A.tif")
        res_dataset = gdal.Open(res_path + str(i) + "_fake_B.tif")
        rec_dataset = gdal.Open(res_path + str(i) + "_rec_B.tif")

        real_arr = real_dataset.ReadAsArray()
        # up_arr = up_dataset.ReadAsArray()[:4]
        res_arr = res_dataset.ReadAsArray()
        rec_arr = rec_dataset.ReadAsArray()

        real_arr = rearrange(real_arr, 'c h w -> h w c')
        # up_arr = rearrange(up_arr, 'c h w -> h w c')
        res_arr = rearrange(res_arr, 'c h w -> h w c')
        rec_arr = rearrange(rec_arr, 'c h w -> h w c')

        # print(real_arr.shape, up_arr.shape, res_arr.shape)

        # up_metric += [list(evaluate(up_arr, real_arr))]
        res_metric += [list(evaluate(res_arr, real_arr))]
        rec_metric += [list(evaluate(rec_arr, real_arr))]

    # pprint(up_metric)
    # pprint(res_metric)

    # up_metric = np.array(up_metric)
    res_metric = np.array(res_metric)
    rec_metric = np.array(rec_metric)
    # print(up_metric.shape, res_metric.shape)

    # up_metric_mean = np.nanmean(up_metric, axis=0)
    res_metric_mean = np.nanmean(res_metric, axis=0)
    rec_metric_mean = np.nanmean(rec_metric, axis=0)

    np.set_printoptions(suppress=True)
    print(["MSE:", 0, "SAM:", 0, "PSNR:", "+&", "SSIM:", 1, "ERGAS:", 0])
    # print(up_metric_mean)
    print(res_metric_mean)
    print(rec_metric_mean)

    # resarr = np.flip(resarr, axis=2)
    # real_netarr = np.flip(real_netarr, axis=2)
    # save_tif(resarr, ref_dataset=res, file_path=".sts_cyclegan_real.tif")

    # realarr = realarr.take([1, 2, 3], axis=0)
    # fakearr = fakearr.take([0, 3, 2], axis=0)

    # realarr = realarr * 255
    # fakearr = fakearr * 255

    # real_netarr = rearrange(real_netarr, 'c h w -> h w c')

    # real = torch.tensor(realarr)
    # fake = torch.tensor(fakearr)

    # real
    # 进行最大最小归一化
    # realarr = min_max_normalize(realarr)
    # fakearr = min_max_normalize(fakearr)

    # mf = gdal.Open("D_result/SELF/result/result0.0455.tif")
    # smif_mf = gdal.Open("D_result/SMIF/fused_output.tif")
    # pca_mf = gdal.Open('D_result/PCA/sharpened_image.tif')
    # ihs_mf = gdal.Open('D_result/IHS-BT-SMIF/fused_image.tif、')
    # pso_mf = gdal.Open('D_result/CS-PSO/CS-PSO_result.tif')
    # glp_mf = gdal.Open('D_result/MYF-GLP/MYF-GLP_result.tif')

    # print(m.RasterXSize, m.RasterYSize)
    # print(smif_mf.RasterXSize, smif_mf.RasterYSize)

    # print("SELF:")

    # MSE1, SAM1, PSNR1, SSIM1, ERGAS1 = evaluate(real_netarr, uparr)
    # pprint(("MSE:", MSE1, "SAM:", SAM1, "PSNR:", PSNR1, "SSIM:", SSIM1, "ERGAS:", ERGAS1))
    # MSE1, SAM1, PSNR1, SSIM1, ERGAS1 = evaluate(real_netarr, resarr)
    # pprint(("MSE:", MSE1, "SAM:", SAM1, "PSNR:", PSNR1, "SSIM:", SSIM1, "ERGAS:", ERGAS1))

#      MSE: 0       SAM: 0      PSNR: +&      SSIM: 1     ERGAS: 0
# [  0.01820271   50.5090472  22.62031522   0.39141423   7.42026673]
# [  0.81615113 146.05684441   0.93741124  -0.16756949   2.55172081]
# [  0.78780675 145.61835581   1.07560186  -0.25137888   2.55369076]
