
import os
import numpy as np
import xarray as xr
from rlxutils import mParallel
from joblib import delayed
import geopandas as gpd
from pyproj import CRS


def extract_chip_from_xarray(z, lat_field, lon_field, chip_geometry):
    is_ascending = lambda x: x[-1]>x[0]        

    coords = np.r_[chip_geometry.boundary.coords]
    minlon, minlat = coords.min(axis=0)
    maxlon, maxlat = coords.max(axis=0)

    if is_ascending(z[lat_field].values):
        lat_range = (minlat, maxlat)
    else:
        lat_range = (maxlat, minlat)

    if is_ascending(z[lon_field].values):
        lon_range = (minlon, maxlon)
    else:
        lon_range = (maxlon, minlon)

    sel_args = {lon_field: slice(*lon_range), lat_field: slice(*lat_range)}
    zz = z.sel(**sel_args).copy()  
    return zz

def extract_chip(source_file, lat_field, lon_field, chip_geometry, chip_identifier, dest_folder, dest_format):
    dest_file = f"{dest_folder}/{chip_identifier}.{dest_format}"
    
    #if os.path.isfile(dest_file):
    #    return
    
    z = xr.open_dataset(source_file)    
    
    zz = extract_chip_from_xarray(z, lat_field, lon_field, chip_geometry)    

    # if crs is not 4326, reproject and extract again to ensure square cropping
    epsg4326 = CRS.from_epsg(4326)
    if zz.rio.crs != epsg4326:
        chip_geometry = gpd.GeoDataFrame([], geometry=[chip_geometry], crs=zz.rio.crs).to_crs(epsg4326).geometry.values[0]
        zz = zz.rio.reproject("EPSG:4326")
        zz = extract_chip_from_xarray(zz, lat_field, lon_field, chip_geometry)    
    
    if dest_format == 'nc':
        zz.to_netcdf(dest_file)
    elif dest_format == 'tif':
        zz = zz.rio.write_crs(epsg4326)
        zz['band_data'].rio.to_raster(dest_file)        
    else:
        raise ValueError("unknown source file format, must be tif or nc (netcdf)") 

def chop(tiles_file, tiles_folder, source_file, lat_field, lon_field, n_jobs=-1):

    z = xr.open_dataset(source_file)    
    print ("reading tiles", flush=True)
    tiles = gpd.read_file(tiles_file)

    if z.rio.crs is not None and z.rio.crs != tiles.crs:
        print ("converting tiles to", z.rio.crs)
        tiles = tiles.to_crs(z.rio.crs)

    if source_file.endswith(".tif") or source_file.endswith(".tiff"):
        dest_format = 'tif'
    elif source_file.endwith(".nc"):
        dest_format = 'nc'
    else:
        raise ValueError("unknown source file format, must be tif or nc (netcdf)") 

    dest_crs = z.rio.crs
    z.close()

    print(f"chopping {len(tiles)} tiles according to {tiles_file}", flush=True)
    mParallel(n_jobs=n_jobs, verbose=30)\
        (delayed(extract_chip)\
                (source_file, lat_field, lon_field, chip.geometry, chip.identifier, tiles_folder, dest_format) \
                    for _ ,chip in tiles.iterrows())
