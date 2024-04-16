import glob
import os

import torch
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from osgeo import gdal
from einops import rearrange

transformsA = [
    # transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.Normalize((0.5,) * 12, (0.5,) * 12),
]
transformsB = [
    # transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.Normalize((0.5,) * 6, (0.5,) * 6),
]


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = transforms.Compose(transformsA)
        self.transform_B = transforms.Compose(transformsB)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index]  # make sure index is within then range
        # if self.opt.serial_batches:  # make sure index is within then range
        #     index_B = index % self.B_size
        # else:  # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index]
        A_dataset = gdal.Open(A_path)
        B_dataset = gdal.Open(B_path)
        A_img = A_dataset.ReadAsArray() / 10000
        B_img = B_dataset.ReadAsArray() / 10000
        A_img = rearrange(A_img, 'c h w -> h w c')
        B_img = rearrange(B_img, 'c h w -> h w c')
        # A_img = torch.tensor(A_img)
        # B_img = torch.tensor(B_img)
        # A_img = Image.fromarray(A_img)
        # B_img = Image.fromarray(B_img)
        # A_img = Image.open(A_path)
        # B_img = Image.open(B_path)
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
