from bmi_topography.topography import Topography
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pathlib import Path
# Sample download

# Need:
# dem_type
# south_coords
# north_coords
# west_coords
# east_coords
# api_key (stored in database)
def save_map(map_name, dem_type, south, north, west, east, api_key):


    user_elevation_map = Topography(dem_type=dem_type, south= south, north=north, west=
    west, east=east, output_format='GTiff', cache_dir="./.hidden", api_key=api_key)
    # code from https://pygis.io/docs/d_raster_crs_intro.html#reproject-a-raster-the-complex-case
    # reproject the map from current to epsg 3857 (web mercator)
    dst_crs = "EPSG:3857"

    with rasterio.open(user_elevation_map.fetch()) as src:
        dst_transform, width, height = calculate_default_transform(
            src.crs,    # source CRS
            dst_crs,    # destination CRS
            src.width,    # column count
            src.height,  # row count
            *src.bounds,  # unpacks outer boundaries (left, bottom, right, top)
        )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
                "nodata": 0
            }
        )
        # Creates temporary directory, change if necessary
        p = Path("./temp")
        p.mkdir(parents=True, exist_ok= True)
        ret_name = "./temp/" + map_name + ".tif"
        with rasterio.open(ret_name, "w", **dst_kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
    
    # Cleans up work
    user_elevation_map.fetch().unlink()
    return ret_name
if __name__ == "__main__":  
    save_map("test", dem_type="AW3D30", south= -36.91320385423351, north=-36.83155361074245, west=
        147.22957375404567, east=147.3532148476227, api_key="51b5286212f68e4dd290e278e914b511")