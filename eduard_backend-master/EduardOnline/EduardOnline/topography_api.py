from django.http import JsonResponse

from Users.src.download import save_map
from Users.models import ElevationMap, CustomUser
from Users.serializers import ElevationMapSerializer, ReliefMapSerializer, GenerateMapSerializer, DownloadMapSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from Users.src.processmodel import convertDispVal
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from Users.models import ReliefMap
from .settings import AWS_S3_ACCESS_KEY_ID, AWS_S3_SECRET_ACCESS_KEY, AWS_STORAGE_BUCKET_NAME
from osgeo import gdal

gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', AWS_S3_SECRET_ACCESS_KEY)
gdal.SetConfigOption('AWS_ACCESS_KEY_ID', AWS_S3_ACCESS_KEY_ID)
import time
import io
import os

# Set to cloudfront distribution name
CLOUDFRONT_URL = 'https://d5xfp7370yhed.cloudfront.net'

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_elevation_maps(request):
    """
    Get all elevation maps belonging to the logged in user

        Returns:
            List of objects with fields:
                user_id (int): ID of the user (primary key)
                file_path (str): file path of the elevation map
                creation_date(str): creation date in yyyy-mm-dd format
                deleted (bool): whether the elevation map file has been deleted or not
    """
    maps = ElevationMap.objects.filter(user_id=request.user)    # get all elevation maps
    result = []
    for map in maps:
        # add the current elevation map to the result
        serializer = ElevationMapSerializer(map)
        result.append(serializer.data)
    return Response(result)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_relief_maps(request):
    """
    Get all relief maps belonging to the logged in user

        Returns:
            List of objects with fields:
                elev_id (int) = ID of the elevation map (primary key)
                user_id (int) = ID of the user (primary key)
                credit_cost (int) = amount of credits required to generate the relief map 
    """
    maps = ReliefMap.objects.filter(user_id=request.user)   # get all maps
    result = []
    for map in maps:
        # add the current relief map to the result
        serializer = ReliefMapSerializer(map)
        result.append(serializer.data)
    return Response(result)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def retrieve_elevation_map(request, map_id):
    """
    Get elevation map by the ID
        Parameters:
            map_id (int): ID of the elevation map

        Returns:
            elev_id (int) = ID of the elevation map (primary key)
            user_id (int) = ID of the user (primary key)
            credit_cost (int) = amount of credits required to generate the relief map
    """
    try:
        # get the elevation map
        map_obj = ElevationMap.objects.get(id=map_id)
        serializer = ElevationMapSerializer(map_obj)
        return Response(serializer.data)
    except ElevationMap.DoesNotExist:
        return Response({"status": "error", "message": "Map not found."}, status=404)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_elevation_map(request, map_id):
    """
    Delete elevation map by the ID from the database
        Parameters:
            map_id (int): ID of the elevation map
    """
    try:
        # delete the elevation map
        map_obj = ElevationMap.objects.get(id=map_id)
        map_obj.delete()
        return Response({"status": "success", "message": "Map deleted successfully."})
    except ElevationMap.DoesNotExist:
        return Response({"status": "error", "message": "Map not found."}, status=404)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_map(request):
    """
    Generate an elevation map based on provided parameters.

        Parameters:
            map_name (str): Name of the downloaded elevation map TIF file
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
            message (str): "Elevation map generated successfully!" if successful
            url (str): URL of the relief map image
    """
    # validate the request body and extract values
    serializer = GenerateMapSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    map_name, macroDisplayedValue, microDisplayedValue, illuminationDisplayedValue, flatAreasAmountDisplayedValue,flatAreasSizeDisplayedValue, terrainTypeDisplayedValue1, terrainTypeDisplayedValue2, nnType, aerialPerpectiveDisplayedValue, contrastDisplayedValue = serializer.data.values()
    terrainTypeDisplayedValue = (terrainTypeDisplayedValue1, terrainTypeDisplayedValue2)

    try:
        # get the paths of the elevation map and model
        savePath = f'/vsis3/{AWS_STORAGE_BUCKET_NAME}/elev/{map_name}'
        modelPath = './Users/src/2022-09-18-r4_4000.pt'

        # Call the convertDispVal function to generate the map
        img = convertDispVal(
            filePath=map_name,
            savePath=savePath,
            modelPath=modelPath,
            macroDisplayedValue=macroDisplayedValue,
            microDisplayedValue=microDisplayedValue,
            illuminationDisplayedValue=illuminationDisplayedValue,
            flatAreasAmountDisplayedValue=flatAreasAmountDisplayedValue,
            flatAreasSizeDisplayedValue=flatAreasSizeDisplayedValue,
            terrainTypeDisplayedValue=terrainTypeDisplayedValue,
            nnType=nnType,
            aerialPerpectiveDisplayedValue=aerialPerpectiveDisplayedValue,
            contrastDisplayedValue=contrastDisplayedValue
        )

        # Save the generated map details to the ReliefMap model
        relief_map = ReliefMap(
            elev_id=ElevationMap.objects.get(file_path=map_name),
            user_id=request.user,
            credit_cost=10,  # Define the credit cost as needed
        )
        relief_map.save()

        # save the image to AWS S3
        path = f'/relief/{str(relief_map.pk)}.jpg'
        i = default_storage.open(path,'w+')
        sfile = io.BytesIO()
        img.save(sfile, format="JPEG")
        i.write(sfile.getvalue())
        i.close()

        return Response({"message": "Elevation map generated successfully!",
                          "url": f'{CLOUDFRONT_URL}/relief/{str(relief_map.pk)}.jpg'}, status=201)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def download_map(request):
    """
    Downloads a relief map from open topography

        Parameters:
            dem_type (str, optional): type of digital elevation map
            south (float): South coordinate 
            north (float): North coordinate
            west (float): West coordinate
            east (float): East coordinate

        Returns:
            status (str): "fail" or "success",
            message (str): "Map downloaded and saved successfully." if the download was successful,
            filename (str): filename of the tif file downloaded
    """
    # get the user's open topography key
    user: CustomUser = request.user
    api_key = user.ot_token

    map_name = str(user.pk) + "_" + str(int(time.time()))   # get the filename used for the downloaded map

    # validate the request body and extract values
    serializer = DownloadMapSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    dem_type, south, north, west, east, amount = serializer.data.values()

    # extracts out the payment amount from the serializer.

    if user.credits < amount:
        return Response({
            "error":"User does not have enough credits"
        }, status=400)

    try:
        # download the map to the server and load its contents
        file_path = save_map(map_name, dem_type, south, north, west, east, api_key)
        with open(file_path, 'rb') as f:
            data = f.read()
            # save the map to AWS S3
        default_storage.save(f'./elev/{map_name}.tif', ContentFile(data))
        # Deletes the map from our internal storage.
        os.remove(file_path)

        # saves the user id and file path
        elev_map = ElevationMap(user_id=user, file_path=f"{map_name}.tif")
        elev_map.save()


        # If everything is successful, then we deduct credits from the user.
        user.credits -= amount
        user.save()
        return JsonResponse({"status": "success", "message": "Map downloaded and saved successfully.", "filename": f'{map_name}.tif'})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
