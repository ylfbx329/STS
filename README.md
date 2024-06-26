# 遥感影像时空谱一体化融合系统

Spatio-Temporal-Spectral Integrated Fusion System of Remote Sensing Images

## 系统简述

具体实现形式为一个能够对遥感影像进行时空谱融合的**Windows桌面应用程序**。

## 技术选型

- 开发语言
    - [ ] C/C++
    - [ ] Java
    - [x] Python 3.10
- GUI框架
    - [ ] Tkinter
    - [x] PyQt
- 打包工具
    - [x] pyinstaller

## 数据

哨兵2：

- 波段（R G B）：B4 B3 B2
- 产品命名规则：
    - https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention
    - 2016年12月6日之前生成的SENTINEL-2 Level-1C产品的旧格式命名已经无法用于官网搜索，GEE得出的图像中使用该PRODUCT_ID的图像可废弃不不用，使用其新命名的图像
- 训练阶段
    - 原始图像
        - S2B_MSIL1C_20171029T170409_N0206_R069_T15RTP_20171029T203238
        - S2B_MSIL1C_20171029T170409_N0206_R069_T15RTN_20171029T203238
    - 标签图像
        - S2A_MSIL1C_20160929T170302_N0204_R069_T15RTP_20160929T171011
        - S2A_MSIL1C_20160929T170302_N0204_R069_T15RTN_20160929T171011
- 数据处理：
    - 软件：下载、选取10m数据、辐射定标、转存为TIFF文件
    - 代码：读取TIFF文件、随机裁剪（200*200）、保存为小型TIFF文件

MODIS

- 波段（R G B）：B01 B04 B03
- 训练阶段
    - MOD09GA.A2017302.h09v05.061.2021288065342
    - MOD09GA.A2017302.h09v06.061.2021288063324
- 数据处理：
    - 软件：下载、重投影、裁剪、转存为TIFF文件（01）、异常值去除01（代码）、拼接01、上采样、转存为TIFF文件
    - 代码：读取TIFF文件、随机裁剪（200*200）、保存为小型TIFF文件

哨兵1：

- 波段（R G B）：VV VH VV

CIA数据集：

- shape = 6, 2040, 1720 = 21,052,800

## 代码框架

数据加载

模型构建

- 模型结构
- 参数初始化
- 损失函数
- 优化器

训练

- 迭代训练
- 模型保存

测试

- 模型加载
- 测试结果可视化



