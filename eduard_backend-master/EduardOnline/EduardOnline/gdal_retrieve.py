from osgeo import gdal
import numpy as np
#### TEST FILE FOR RETRIEVING DATA


import numpy.typing as npt
from settings import AWS_S3_ACCESS_KEY_ID, AWS_S3_SECRET_ACCESS_KEY, AWS_STORAGE_BUCKET_NAME
gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', AWS_S3_SECRET_ACCESS_KEY)
gdal.SetConfigOption('AWS_ACCESS_KEY_ID', AWS_S3_ACCESS_KEY_ID)


def load_tif_from_S3(path: str)-> npt.NDArray:
    """Loads a tif from an S3 bucket and converts it to a numpy array. 

    Args:
        path (str): Path of string

    Returns:
        npt.ArrayLike: First band from the array.
    """    
    location = f'/vsis3/{AWS_STORAGE_BUCKET_NAME}/{path}'
    dataset = gdal.Open(location, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)

    return np.array(band.ReadAsArray())

if __name__ == "__main__":
    load_tif_from_S3("1_1695270690.tif")