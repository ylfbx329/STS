from osgeo import gdal

ds = gdal.Open(r'D:\useful\program\rs\STS\data\mod_01.tif')
print(ds.GetGeoTransform())
print(ds.GetProjection())