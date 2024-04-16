# -*- coding:UTF-8 -*-
import arcpy
'''
    1、gdal 模块可以实现，但是要通过xy坐标值才能获取相应的 pixel value，过程较为麻烦
    2、arcpy 模块里面有 GetRasterProperties_management方法，可以读取不同波段的不同计算方法
    3、直接在arcgis -> 数据管理工具 -> 栅格 -> 栅格属性 -> 获取栅格数据 这个工具不知道为什么在桌面端里面计算的值都是一个波段的值，似乎
        选择波段不起作用，可能是破解版的BUG
'''
path = r'输入tif的路径'
bands = 7  # 输入你影像的波段数量
# 下面函数的功能和方法一【计算统计数据】功能一样
arcpy.CalculateStatistics_management(path, '', '', '', '', '')
for i in range(1, bands + 1 ):
	# 第二个参数输入MEAN是求平均值，STD是求方差。还有很多个功能，这时候的arcpy真香
    result = arcpy.GetRasterProperties_management(path, 'MEAN', 'Band_{}'.format(i))
    mean = result.getOutput(0)
    print mean

from osgeo import gdal
gdalObject = gdal.Open('路径')
# i 为波段
gdalObject.GetRasterBand(i).ComputeStatistics(false, out tmpDou, out tmpDou, out tmpDou, out tmpDou, null, null)
