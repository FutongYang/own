from osgeo import gdal
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import scipy as sp
import torch
from .model_plain import UNet
import numpy.typing as npt
from math import ceil, floor
from EduardOnline.settings import AWS_S3_ACCESS_KEY_ID, AWS_S3_SECRET_ACCESS_KEY
from django.core.files.storage import default_storage
import io

from PIL import Image
PADDED_AMT = 50
IMG_SIZE_PASS_MODEL = 512
IMG_SIZE_OUT_MODEL = IMG_SIZE_PASS_MODEL - 2 * PADDED_AMT

VF1 = 0.8925781
VF2 = 0.8408203
VF3 = 0.79541016

LOOKUP1 = [76,76,77,77,78,78,79,79,80,80,81,81,82,82,82,83,83,84,84,85,85,86,86,
           87,87,88,88,88,89,89,90,90,91,91,92,92,93,93,94,94,95,95,96,96,97,97,
           98,98,98,99,99,100,100,101,101,102,102,103,103,104,104,105,105,106,
           106,107,107,108,109,109,110,110,111,111,112,112,113,113,114,114,115,
           115,116,116,117,118,118,119,119,120,120,121,121,122,123,123,124,124,
           125,125,126,127,127,128,128,129,129,130,131,131,132,133,133,134,134,
           135,136,136,137,137,138,139,139,140,141,141,142,143,143,144,145,145,
           146,147,147,148,149,149,150,151,151,152,153,153,154,155,156,156,157,
           158,159,159,160,161,161,162,163,164,165,165,166,167,168,168,169,170,
           171,172,172,173,174,175,176,176,177,178,179,180,181,181,182,183,184,
           185,186,187,187,188,189,190,191,192,193,194,195,196,196,197,198,199,
           200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,
           217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,
           234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,
           251,252,253,254,255]
LOOKUP1 = np.array(LOOKUP1)

LOOKUP2 = [100,100,100,101,101,101,101,102,102,102,102,103,103,103,103,103,104,
           104,104,104,105,105,105,105,106,106,106,106,107,107,107,107,108,108,
           108,108,109,109,109,109,110,110,110,110,111,111,111,112,112,112,112,
           113,113,113,113,114,114,114,115,115,115,116,116,116,116,117,117,117,
           118,118,118,119,119,119,120,120,120,121,121,121,122,122,123,123,123,
           124,124,124,125,125,126,126,126,127,127,128,128,128,129,129,130,130,
           131,131,132,132,132,133,133,134,134,135,135,136,136,137,137,138,138,
           139,139,140,140,141,141,142,142,143,144,144,145,145,146,146,147,148,
           148,149,149,150,151,151,152,153,153,154,155,155,156,157,157,158,159,
           159,160,161,162,162,163,164,164,165,166,167,168,168,169,170,171,171,
           172,173,174,175,176,176,177,178,179,180,181,182,183,183,184,185,186,
           187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,
           204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,
           221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,
           238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,
           255]

LOOKUP2 = np.array(LOOKUP2)

LOOKUP3 = [67,68,68,69,69,70,70,71,72,72,73,73,74,75,75,76,76,77,77,78,79,79,80,
           80,81,82,82,83,83,84,84,85,86,86,87,87,88,89,89,90,90,91,91,92,93,93,
           94,94,95,96,96,97,97,98,99,99,100,100,101,102,102,103,103,104,105,
           105,106,106,107,108,108,109,110,110,111,111,112,113,113,114,115,115,
           116,116,117,118,118,119,120,120,121,121,122,123,123,124,125,125,126,
           127,127,128,129,129,130,130,131,132,132,133,134,134,135,136,136,137,
           138,138,139,140,141,141,142,143,143,144,145,145,146,147,147,148,149,
           150,150,151,152,152,153,154,154,155,156,157,157,158,159,160,160,161,
           162,162,163,164,165,165,166,167,168,168,169,170,171,172,172,173,174,
           175,175,176,177,178,179,179,180,181,182,182,183,184,185,186,186,187,
           188,189,190,191,191,192,193,194,195,196,196,197,198,199,200,201,202,
           202,203,204,205,206,207,208,209,209,210,211,212,213,214,215,216,217,
           218,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,
           234,235,236,237,238,239,240,241,242,243,244,245,246,248,249,250,251,
           252,253,254,255]

LOOKUP3 = np.array(LOOKUP3)

class transformedArray:
    """New container class which stores initial array shape
    """    
    def __init__(self, array:npt.NDArray):
        """Initialises creation of transformed (rotated, padded etc) array. Can be manipulated to store other items.

        Args:
            array (npt.NDArray): Initial unrotated array.
        """        
        self.arr = array
        self.l, self.h = array.shape
    
    def set_arr(self, array:npt.NDArray):
        self.arr = array




def load_model(modelPath, removePadding = False, deviceStr = "cpu") -> UNet:
    """Loads model from the final checkpoint.

    Args:
        modelPath (_type_): Type of model to load from.
        removePadding (bool, optional): Removed padding from the final input
            ; is 50 units. If false, then padding is removed.
            If true, padding is not removed. 
            Defaults to False.
        deviceStr (str, optional): Either "cpu" or "gpu".
            Defaults to "cpu".

    Returns:
        UNet: A Unet model of the relief shader. 
    """    
    
    device = torch.device(deviceStr)
    model = UNet((256, 256, 1), PADDED_AMT, remove_padding=removePadding)
    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
def load_tif_as_arr(path: str)-> npt.NDArray[np.float32]:
    """Loads a tif and converts it to a nump array. 

    Args:
        path (str): Path of string

    Returns:
        npt.ArrayLike: First band from the array.
    """    
    gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', AWS_S3_SECRET_ACCESS_KEY)
    gdal.SetConfigOption('AWS_ACCESS_KEY_ID', AWS_S3_ACCESS_KEY_ID)
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)

    return np.array(band.ReadAsArray())

def pad_50(array:npt.NDArray) -> npt.NDArray:
    """Pads the array by 50 units with a flipped copy of the array on the border as padding

    Args:
        array (npt.NDArray): Unpadded input input

    Returns:
        npt.NDArray: Unpadded input
    """    
    padded_arr = np.pad(array, pad_width=PADDED_AMT, mode='symmetric')
    # pads array out with symmetric values
    return padded_arr

def unpad_50(array: npt.NDArray) -> npt.NDArray:
    """Unpads the array

    Args:
        array (npt.NDArray): Padded array

    Returns:
        npt.NDArray: Unpadded array
    """    
    l,w = array.shape
    return array[PADDED_AMT:l - PADDED_AMT,PADDED_AMT:w - PADDED_AMT]

def prepare_input(arr: npt.NDArray) -> torch.Tensor:
    """Generates new array to be parsed by UNet model

    Args:
        arr (npt.NDArray): array to be parsed by model

    Returns:
        torch.Tensor: Tensor model of array.
    """    
    new_arr = [[arr]]
    new_arr = np.array(new_arr)
    return torch.from_numpy(new_arr).type(torch.float32)


def run_model(model:UNet,prepared_input: torch.Tensor ) -> npt.NDArray:
    """Runs the model on the input and returns an array with dimensions
    (x,y) which is 2^n or 2^n - 100 depending on whether borders are removed.

    Args:
        model (UNet): UNet model to run input on
        prepared_input (torch.Tensor): Tensor input

    Returns:
        npt.NDArray: Numpy array input.
    """    
    out = model(prepared_input)
    return out.detach().numpy()[0][0]

def gaussian_blur(array: npt.NDArray, generalizationDetails= 1.0)->npt.NDArray:
    """Blurs array with a gaussian blur

    Args:
        array (npt.NDArray): Array to blur
        generalizationDetails (float, optional): Amount to blur by. Defaults to 1.0.

    Returns:
        npt.NDArray: Blurred array by the generalised amount.
    """    
    if generalizationDetails == 1.0:
        return array
    else:
        n = ceil(generalizationDetails) * 2 + 1
        return cv.GaussianBlur(array, (n,n), generalizationDetails/3)

def rotate_array(array:npt.NDArray, lightRotation = 0) -> transformedArray:
    """Rotates array and returns object with rotated array and initial length and width.

    Args:
        array (npt.NDArray): Initial array
        lightRotation (int, optional): Amount to rotate by anticlockwise. Defaults to 0.

    Returns:
        RotatedArray: _description_
    """    
    newArray = transformedArray(array)
    newArray.arr = sp.ndimage.rotate(array, lightRotation, reshape= True, mode = 'reflect')

    return newArray

def unrotate_array(rotArr: transformedArray, lightRotation = 0) -> npt.NDArray:
    """Unrotates a 'rotated' array.

    Args:
        rotArr (RotatedArray): Rotated 2D array with the initial length and width
        lightRotation (int, optional): Angle to rotate the array clockwise by. Defaults to 0.

    Returns:
        npt.NDArray: Unrotated 2D array
    """    
    newArray = sp.ndimage.rotate(rotArr.arr, -lightRotation,reshape= True) # rotated, but at what cost?
    if lightRotation in [0, 90, 180, 270]:
        return newArray
    else:
        x,y = newArray.shape
        centx,centy = x//2, y//2 # Centre is preserved for every circle
        distx, disty = rotArr.l//2, rotArr.h//2
        return newArray[centx - distx:centx + distx, centy - disty: centy + disty]
    # return newArray

def rescale_arr(arr:npt.NDArray) -> npt.NDArray:
    """
    Scales array to be between 0 and 1.

    Args:
        arr (npt.NDArray): Initial 2d array, can be floats or integers

    Returns:
        npt.NDArray: Returns 2d float array
    """    
    minArr = np.min(arr)
    maxArr = np.max(arr)

    return (arr - minArr)/(maxArr - minArr) # type: ignore

def downsample_arr(arr: npt.NDArray, generalization = 1.0) -> npt.NDArray:
    """Downsamples array with cubic function

    Args:
        arr (npt.NDArray): initial array
        generalization (int, optional): Fraction to shrink image by. Defaults to 1.

    Returns:
        npt.NDArray: Downsampled array
    """    
    return cv.resize(arr, None, fx=1/generalization,
                    fy=1/generalization, interpolation=cv.INTER_CUBIC)

def upsample_arr(arr: npt.NDArray, generalization = 1.0) -> npt.NDArray:
    """Upsamples array with inter cubic function. Pair this function with the
    above one.

    Args:
        arr (npt.NDArray): Array to upsample by
        generalization (int, optional): Mutiple to upscale by. Defaults to 1.

    Returns:
        npt.NDArray: Upsampled array.
    """    
    return cv.resize(arr, None, fx=generalization,
                    fy=generalization, interpolation=cv.INTER_CUBIC)
# model scale

def aerial_perspective(reliefMap: npt.NDArray, elevModel: npt.NDArray, aerialPerspective = 0.0, nType:int = 1) -> npt.NDArray:
    """Calculates the aerial perspective multiplier given the original elevation map.
    reliefMap and elevModel should have the same dimensions, roughly. 
    Unscale relief map by n if original map is scaled by a factor of n. 

    Args:
        reliefMap (npt.NDArray): Relief map, scaled to be between 0 and 1.
        elevModel (npt.NDArray): Elevation model- shape of elevModel and reliefMap should be (nearly)identical.
        elevModel should be scaled to be between 0 and 1. 
        aerialPerspective (float, optional): Amount to scale the model by. Defaults to 0.
        nType (int, optional): The type of model used. Defaults to 1.

    Returns:
        npt.NDArray: Scaled relief map model with every element scaled with final map.
        All values are between 0 and 1. 
    """    
    x1,y1 = reliefMap.shape
    x2, y2 = elevModel.shape
    x = min(x1, x2) 
    y = min(y1, y2)
    z = rescale_arr(elevModel)[:x,:y] # shaves off the edges to make the relief map and elev map align
    v = reliefMap[:x,:y]
    vf = VF1
    if nType == 1:
        vf = VF1
    elif nType == 2:
        vf = VF2
    elif nType == 3:
        vf = VF3

    v_ = aerialPerspective * vf + aerialPerspective * v
    squarez = np.square(z)
    multreturn = np.multiply(squarez, (v - v_))
    return np.add(multreturn, v_)

def contrast_map(reliefMap: npt.NDArray, slopeDarkness = 0.0, nType: int = 1) -> npt.NDArray:
    """Generates a more contrasted image.

    Args:
        reliefMap (npt.NDArray): 2D Numpy array where all values between 0 and 1. 
        slopeDarkness (float, optional): Darkness of slope between 0 and 1 where 1 is the darkest slope. Defaults to 0.0.
        nType (int, optional): The number of the neural network. Defaults to 1.

    Returns:
        npt.NDArray: A uint8 array where every value is scaled between 0 and 255.
    """
    lookupTable = LOOKUP1
    if nType == 1:
        lookupTable = LOOKUP1
    elif nType == 2:
        lookupTable = LOOKUP2
    elif nType == 3:
        lookupTable = LOOKUP3
    
    lookupArr = np.zeros(256)
    for index, val in enumerate(lookupTable):
        blend = val*(1-slopeDarkness) + index * slopeDarkness
        if blend > 255:
            blend = 255
        if blend < 0:
            blend = 0
        blend = np.rint(blend)
        lookupArr[index] = blend
    # relief map is between 0 and 1
    rescaledreliefMap = reliefMap * 255
    # code from https://www.statology.org/numpy-array-to-int/
    arr8bit = (np.floor(rescaledreliefMap)).astype(int)
    clipped_arr = np.clip(arr8bit, 0, 255)
    return lookupArr[clipped_arr]

def generate_image(imgPath: str, reliefMap: npt.NDArray):
    """Generates and saves an image using pillow given the string path above.

    Args:
        imgPath (str): Path to image, no extension necessary.
        reliefMap (npt.NDArray): The uint8 relief map.
    """    
    img = Image.fromarray(np.uint8(reliefMap), 'L')
    return img

def resize_square(array: npt.NDArray) -> transformedArray:
    """Resizes the array to the smallest power of 2 that contains the array + 100 pixels on the edges
    and restricts it to n - 100 so we can pad the edges
    We fill the extra space with random noise.

    Args:
        array (npt.NDArray): _description_

    Returns:
        transformedArray: _description_
    """    
    arrToChange = transformedArray(array)
    x,y = array.shape
    maxsize = max(arrToChange.arr.shape) + 2*PADDED_AMT
    squaresize = np.power(2, ceil(np.log2(maxsize))) - 2 * PADDED_AMT
    # size of smallest 2^n square that fits maxsize - 100
    squaresize = int(squaresize)
    resized_arr = np.zeros((squaresize, squaresize))
    x_, y_ = resized_arr[x:2 * x, :y].shape
    resized_arr[x:2 * x, :y] = np.flip(array[x - x_:, y - y_: ], 0)
    x_, y_ = resized_arr[:x, y:2*y].shape
    resized_arr[:x, y:2*y] = np.flip(array[x - x_:, y - y_: ], 1)
    resized_arr[:x, :y] = array

    arrToChange.set_arr(resized_arr)
    return arrToChange

def unresize_square(transArr: transformedArray) -> npt.NDArray:
    """Unresizes the array given a transformed array.

    Args:
        transArr (transformedArray): The transformed array

    Returns:
        npt.NDArray: An array with the extra information removed. 
    """    
    x,y = transArr.l, transArr.h

    return transArr.arr[:x, :y]

def elevationMap(arr: npt.NDArray, elevation_min:float, elevation_max:float) -> npt.NDArray:
    scaled_arr = rescale_arr(arr)
    return scaled_arr/elevation_max + elevation_min


    
    

if __name__ == "__main__":
    path = ".bmi_topography/AW3D30_-36.91320385423351_147.22957375404567_-36.83155361074245_147.3532148476227.tif"
    model_path = "2022-09-18-r4_4000.pt"
    arr = load_tif_as_arr(path)
    next_arr = rescale_arr(arr)
    print(arr.shape)
    rotate_by = 90
    new_arr = rotate_array(next_arr, rotate_by)
    newer_arr = unrotate_array(new_arr, rotate_by)
    print(newer_arr.shape)
    plt.subplot(131), plt.imshow(next_arr), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(new_arr.arr), plt.title('Rotated')

    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(newer_arr), plt.title('Unrotated')
    plt.xticks([]), plt.yticks([])
    plt.show()

    shrunken_arr = downsample_arr(next_arr, 2)
    unshrunken_arr = upsample_arr(shrunken_arr, 2)
    # plt.subplot(131), plt.imshow(next_arr), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.imshow(shrunken_arr), plt.title('Shrunken')

    # plt.xticks([]), plt.yticks([])
    # plt.subplot(133), plt.imshow(unshrunken_arr), plt.title('Unshrunken')
    # plt.xticks([]), plt.yticks([])
    # plt.show()



    cropped_arr = rescale_arr( next_arr[:256, :256])
    centre_arr = next_arr[50:206, 50:206]
    model = load_model(model_path, removePadding=True)
    output = run_model(model, prepared_input= prepare_input(cropped_arr) )

    aerial = 0.5
    aerialisedOut = aerial_perspective(output, cropped_arr, aerial)
    # plt.subplot(131), plt.imshow(cropped_arr), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.imshow(output), plt.title('Passed through NN')

    # plt.xticks([]), plt.yticks([])
    # plt.subplot(133), plt.imshow(aerialisedOut), plt.title('Aerialised with 0.5')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    # x = 50
    # y = 50
    # z = cropped_arr[x,y]
    # v = output[x,y]
    # v_ = aerial * VF1 + (1-aerial) * v
    # v__ = (v - v_) * z * z + v_
    # print(f'actual out: {v__}\n aerialisedOut: {aerialisedOut[x,y]}')

    # plt.imshow(cropped_arr)
    # plt.imshow(output, cmap='jet', alpha=0.2)
    # plt.show()

    # plt.imshow(cropped_arr)
    # plt.imshow(aerialisedOut, cmap='jet', alpha=0.2)
    # plt.show()

    nocontrast = contrast_map(aerialisedOut, 0, 1)
    contrast = contrast_map(aerialisedOut, 1,1)
    # plt.subplot(131), plt.imshow(cropped_arr), plt.title('Original elev')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.imshow(aerialisedOut * 255), plt.title('Aerialised')

    # plt.xticks([]), plt.yticks([])
    # plt.subplot(133), plt.imshow(contrast), plt.title('Contrast of 0')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    # generate_image("contrast", contrast)
    # generate_image("no_contrast", nocontrast)

    model = load_model(model_path, removePadding=True)
    new_arr = resize_square(next_arr)
    newer_arr = pad_50(new_arr.arr)
    input = prepare_input(newer_arr)
    rel_map = run_model(model, input)
    unpadded_arr = unpad_50(rel_map)
    new_arr.set_arr(unpadded_arr)
    final_img = unresize_square(new_arr)

    # plt.subplot(141), plt.imshow(next_arr), plt.title('Original array')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(142), plt.imshow(newer_arr), plt.title('Resized image')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(143), plt.imshow(unpadded_arr), plt.title('Unpadded array')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(144), plt.imshow(final_img), plt.title('Final array with image resized correctly')
    # plt.xticks([]), plt.yticks([])
    plt.imshow(next_arr, cmap="BuPu")
    plt.imshow(aerial_perspective(final_img, next_arr, aerialPerspective=0.5), cmap='gray', alpha=1)
    plt.show()


