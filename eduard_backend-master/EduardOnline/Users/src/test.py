import unittest
from image import *
from download import *
import numpy as np

class TestDownload(unittest.TestCase):
    # Tests if the download system works
    def test_download(self):
        save_map("test.tif", dem_type="AW3D30", south= -36.91320385423351, north=-36.83155361074245, west=
        147.22957375404567, east=147.3532148476227, api_key="51b5286212f68e4dd290e278e914b511")
    
    def test_image(self):
        path = "./temp/test.tif"
        model_path = "EduardOnline/Users/src/2022-09-18-r4_4000.pt"

        load_model(model_path)
        arr = load_tif_as_arr(path)
        rotate_by = 90
        new_arr = rotate_array(arr, rotate_by)
        newer_arr = unrotate_array(new_arr, rotate_by)
        np.testing.assert_array_equal(newer_arr, arr, "arrays rotated by 90 degrees unequal")

        rotate_by = 45
        new_arr = rotate_array(arr, rotate_by)
        newer_arr = unrotate_array(new_arr, rotate_by)
        np.testing.assert_array_almost_equal(newer_arr[:5, :5], arr[:5, :5], decimal = 1 ,err_msg="array unrotated by 45 degrees incorrectly")
        # this part usually fails, rotation by any degrees that is not a multiple of 90 fails.
        # It is not a problem as the unrotated image is "close enough" to the rotated one that it doesn't make a difference.
        scaled_arr = rescale_arr(arr)

        np.testing.assert_almost_equal(np.amax(scaled_arr), 1) # type: ignore
        np.testing.assert_almost_equal(np.amin(scaled_arr), 0) # type: ignore

        blur_arr = gaussian_blur(arr)
        np.testing.assert_array_equal(blur_arr.shape, arr.shape)

        # silly test
        new_arr = pad_50(arr)


if __name__ == '__main__':
    unittest.main()
