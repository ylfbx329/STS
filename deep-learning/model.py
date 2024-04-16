import torch
from torch import nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        out_features = 64
        self.in_block = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, out_features, kernel_size=7, stride=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features

        self.encoder = []
        for _ in range(2):
            out_features *= 2
            self.encoder += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        self.res_list = nn.ModuleList([ResidualBlock(in_features) for _ in range(6)])

        self.decoder = []
        for _ in range(2):
            out_features //= 2
            self.decoder += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        self.decoder[-3].output_padding = 1

        self.out_block = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, out_channels, kernel_size=7, stride=1),
            nn.Tanh()
        ]

        self.in_block = nn.Sequential(*self.in_block)
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        self.out_block = nn.Sequential(*self.out_block)

    def forward(self, x):
        x = self.in_block(x)
        x = self.encoder(x)
        for res in self.res_list:
            x = res(x)
        x = self.decoder(x)
        x = self.out_block(x)
        return x


# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        """
        input_shape = (c,h,w)
        """
        super().__init__()

        in_channels = input_shape[0]
        self.output_shape = (input_shape[1] // 8 - 8, input_shape[2] // 8 - 8)

        out_features = 64
        self.in_block = [
            nn.Conv2d(in_channels, out_features, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True)
        ]
        in_features = out_features

        self.blocks = []
        for _ in range(2):
            out_features *= 2
            self.blocks += [
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        self.blocks += [
            nn.Conv2d(in_features, in_features, kernel_size=4, stride=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
        ]

        self.out_block = [
            nn.Conv2d(in_features, 1, kernel_size=4, stride=1),
            nn.Sigmoid()
        ]

        self.in_block = nn.Sequential(*self.in_block)
        self.blocks = nn.Sequential(*self.blocks)
        self.out_block = nn.Sequential(*self.out_block)

    def forward(self, x):
        x = self.in_block(x)
        x = self.blocks(x)
        x = self.out_block(x)
        return x
