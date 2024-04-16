import numpy as np
from osgeo import gdal

x = [105, 105, 105, 105, 106, 106, 106, 106]
y = [166, 167, 168, 169, 167, 168, 169, 170]


def outlier(tiff_path):
    dataset = gdal.Open(tiff_path, gdal.GA_Update)
    # 获取波段数量
    num_bands = dataset.RasterCount

    for band_index in range(1, num_bands + 1):
        # 读取当前波段的数据
        band = dataset.GetRasterBand(band_index)
        data = band.ReadAsArray()

        # 获取当前波段平均值作为填充
        mean = np.mean(data[np.where((data > -100) & (data < 100))])

        # 将目标值修改为新的值
        for xi, yi in zip(x, y):
            data[xi, yi] = mean

        # 将修改后的数据写回到波段
        band.WriteArray(data)

    # 关闭数据集，保存修改
    del dataset


outlier(r"D:\useful\program\rs\data\tmpout\tiffout\mod_01_2.tif")
