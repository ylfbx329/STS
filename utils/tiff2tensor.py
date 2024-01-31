from osgeo import gdal
from torchvision.transforms import functional as F


class ReadTiff(object):
    def readTif_to_Ndarray(self, img_path):
        dataset = gdal.Open(img_path)
        # 获得矩阵的列数
        self.width = dataset.RasterXSize
        # 栅格矩阵的行数
        self.height = dataset.RasterYSize
        # 获得数据
        self.data = dataset.ReadAsArray(0, 0, self.width, self.height)
        return self.data

    def Ndarray_to_Tensor(self, ndarray)
        '''
        这里的permute函数是变换维度的函数，我输入的tif维度是[7,448,448]，但是
        gdal读取的时候是[448,7,448]和cv2的读取方式相似，所以这里用permute(1,0,2)把
        维度转换为正常的[C,W,H]
        '''
        F.to_tensor(ndarray).permute(1, 0, 2)
        return Tensor
