from osgeo import gdal


def save_tif(array, upper_left_x=None, upper_left_y=None, dtype=None, ref_dataset=None, file_path=None):
    """
    array.shape = (channel, height, width)
    :param array:
    :param upper_left_x:
    :param upper_left_y:
    :param dtype:
    :param ref_dataset:
    :param file_path:
    :return:
    """
    if dtype is None:
        dtype = ref_dataset.GetRasterBand(1).DataType

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(file_path,
                            xsize=array.shape[1], ysize=array.shape[2], bands=array.shape[0],
                            eType=dtype)
    dataset.SetProjection(ref_dataset.GetProjection())
    ref_trans = ref_dataset.GetGeoTransform()
    print(ref_trans)
    if upper_left_x is not None and upper_left_y is not None:
        trans = (
            ref_trans[0] + upper_left_x * ref_trans[1],
            ref_trans[1],
            ref_trans[2],
            ref_trans[3] + upper_left_y * ref_trans[5],
            ref_trans[4],
            ref_trans[5])
        print(trans)
        dataset.SetGeoTransform(trans)
    else:
        dataset.SetGeoTransform(ref_trans)

    if array.shape[0] == 1:
        dataset.GetRasterBand(1).WriteArray(array)
    else:
        for i in range(array.shape[0]):
            dataset.GetRasterBand(i + 1).WriteArray(array[i])

    dataset.FlushCache()
    del dataset
