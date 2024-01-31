import os
import glob

from osgeo import gdal
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class STSDataset(Dataset):
    def __init__(self, root, _transforms=None, mode="train"):
        """
        实例化时运行一次
        :param root:
        :param transforms:
        :param mode:
        """
        self.files_A = sorted(glob.glob(os.path.join(root, "%sA/*.*" % mode)))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB/*.*" % mode)))
        self.transform_A = transforms.Compose(_transforms[0])
        self.transform_B = transforms.Compose(_transforms[1])

    def __len__(self):
        """
        返回样本总数
        :return:
        """
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        """
        根据索引获取一个样本-标签对
        :param index:
        :return:
        """
        data_A = gdal.Open(self.files_A[index])
        data_B = gdal.Open(self.files_B[index])
        arr_A = rearrange(data_A.ReadAsArray(), "c h w -> h w c")
        arr_B = rearrange(data_B.ReadAsArray(), "c h w -> h w c")
        item_A = self.transform_A(arr_A)
        item_B = self.transform_B(arr_B)
        del data_A, data_B, arr_A, arr_B
        return {"A": item_A, "B": item_B}
