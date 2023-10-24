from .image import *
def convertDispVal( filePath: str,
                    savePath: str,
                    modelPath: str,
                    macroDisplayedValue = 0,
                    microDisplayedValue= 0, 
                    illuminationDisplayedValue = 0, 
                    flatAreasAmountDisplayedValue = 0, 
                    flatAreasSizeDisplayedValue = 0,
                    terrainTypeDisplayedValue:tuple[int, int] = (0,100), 
                    nnType = 1, 
                    aerialPerpectiveDisplayedValue= 100,
                    contrastDisplayedValue = 0):
    """This function converts all values from between 0 to 100 (usually)
    to a value between 0 and 1.

    Args:
        filePath (str): The relative path of the file where we want to load from (Deprecated)
        savePath (str): The full path from which we want to load from (Use this)
        modelPath (str): Full path of NN model
        macroDisplayedValue (int, optional): Macro desplayed value for downsampling (between 0 and 100). Defaults to 0.
        microDisplayedValue (int, optional): Display value for Gaussian blur. Defaults to 0.
        illuminationDisplayedValue (int, optional): Clockwise angle (in degrees) of the light, 0 is at the top left. Defaults to 0.
        flatAreasAmountDisplayedValue (int, optional): Flat areas amount. Does nothing. . Defaults to 0.
        flatAreasSizeDisplayedValue (int, optional): Flat areas size. Does nothing. Defaults to 0.
        terrainTypeDisplayedValue (tuple[int, int], optional): Scales the values in the array to be between the two tuple numbers. Defaults to (0,100).
        nnType (int, optional): The type of neural network-current value is 1. Defaults to 1.
        aerialPerpectiveDisplayedValue (int, optional): Scales the perspective depending on the elevation map. Defaults to 100.
        contrastDisplayedValue (int, optional): Scales the contrast, making the images lighter. Defaults to 0.

    Returns:
        Image: The relief image, as an array.
    """    
    
    assert 0 <= macroDisplayedValue <= 100
    generalization = macroDisplayedValue * 9 / 100 + 1 # resampling
    
    assert 0 <= microDisplayedValue <= 100
    generalizationDetails = microDisplayedValue * 9 / 100 + 1 # gaussian blur

    assert 0 <= illuminationDisplayedValue < 360
    lightRotation = illuminationDisplayedValue

    assert 0 <= flatAreasAmountDisplayedValue <= 100
    assert 0 <= flatAreasSizeDisplayedValue <= 100

    assert 0 <= terrainTypeDisplayedValue[0] <= terrainTypeDisplayedValue[1] <= 100
    elevationRange = (terrainTypeDisplayedValue[0]/100, terrainTypeDisplayedValue[1]/100)
    assert 0 <= aerialPerpectiveDisplayedValue <= 100
    aerialPerpective = aerialPerpectiveDisplayedValue/100

    assert 0<= contrastDisplayedValue <= 100
    slopeDarkness = contrastDisplayedValue/100

    return runProcess(filePath, savePath, modelPath, nnType, generalization, generalizationDetails, lightRotation, elevationRange, aerialPerpective, slopeDarkness)

def runProcess(filePath, savePath, modelPath, nnType = 1, generalization = 1.0, generalizationDetails = 1.0, lightRotation = 0, elevationRange = (0.0, 1.0), aerialPerpective = 0.0, slopeDarkness = 0.0):
    """This is the pipeline from converting a relief model into an image.

    Args:
        filePath (str): Path to image, deprecated
        savePath (str): Path to elevation model
        modelPath (str): Path to NN model
        nnType (int, optional): Type of neural network. Defaults to 1.
        generalization (float, optional): Fraction to shrink down by. Defaults to 1.0.
        generalizationDetails (float, optional): Fraction to blur by. Defaults to 1.0.
        lightRotation (int, optional): Amount the light is turned counterclockwise. Defaults to 0.
        elevationRange (tuple, optional): Places the elevation values between the two ranges. Defaults to (0.0, 1.0).
        aerialPerpective (float, optional): Changes the elevation multiplier. Defaults to 0.0.
        slopeDarkness (float, optional): Changes how much darker the image is. Defaults to 0.0.

    Returns:
        Image: An 8-bit image of the relief shading
    """    
    # Loads the model
    model = load_model(modelPath)
    image = load_tif_as_arr(savePath)
    # print(image.shape)

    # Does the preprocessing steps to pass into model
    scaled_img = rescale_arr(image)
    # resamples array
    resampled_arr = downsample_arr(scaled_img, generalization)

    # blurs array
    blurred_arr = gaussian_blur(resampled_arr, generalizationDetails)

    # rotate array
    rotated_array = rotate_array(blurred_arr, lightRotation)
    # print(blurred_arr.shape)
    # print(rotated_array.arr.shape)
    # rescales array
    recaled_arr = elevationMap(rotated_array.arr, elevation_min=elevationRange[0], elevation_max=elevationRange[1] )
    # resizes container
    resizeContainer = resize_square(recaled_arr)
    resized_arr = resizeContainer.arr
    # pads array
    padded_arr = pad_50(resized_arr)
    # plt.imshow(padded_arr)
    #plt.show()
    tensor_model = prepare_input(padded_arr)

    # Pass to model
    out = run_model(model, tensor_model)

    # Does post-processing model, reverses steps in pre-processing model
    resizeContainer.set_arr(out)
    # unresizes square
    cut_out = unresize_square(resizeContainer)
    # unrotate array
    rotated_array.set_arr(cut_out)
    unrotated_arr = unrotate_array(rotated_array, lightRotation=lightRotation)
    # upsamples arr
    desampled_arr = upsample_arr(unrotated_arr, generalization)
    # Aerialises model
    aerialised_out = aerial_perspective(desampled_arr, scaled_img, aerialPerpective)

    # array scaled to be between 0 and 1    
    a = max(np.max(aerialised_out),1) # type: ignore

    scaled_out = aerialised_out/a

    # prepares model for saving; creates map with uint8 values from values b/w 0 and 1
    contrastOut = contrast_map(scaled_out, slopeDarkness, nnType)
    # saves model
    return generate_image(savePath, contrastOut)



if __name__ == "__main__":
    runProcess('./temp/test.tif',
                   'test0',
                   './EduardOnline/NeuralNetwork/src/2022-09-18-r4_4000.pt', lightRotation=0, aerialPerpective=0.5)
    