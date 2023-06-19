
import os
import numpy as np
import xarray as xr
from rlxutils import mParallel
from joblib import delayed
import geopandas as gpd

def extract_chip(source_file, lat_field, lon_field, chip_geometry, chip_identifier, dest_folder):
    dest_file = f"{dest_folder}/{chip_identifier}.nc"
    if os.path.isfile(dest_file):
        return
    
    z = xr.open_dataset(source_file)    
    
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

    zz = z.sel(lon=slice(*lon_range), lat=slice(*lat_range)).copy()  
    zz.to_netcdf(dest_file)


def chop(tiles_file, tiles_folder, source_file, lat_field, lon_field, n_jobs=-1):
    print ("reading tiles", flush=True)
    tiles = gpd.read_file(tiles_file)
    print(f"chopping {len(tiles)} tiles according to {tiles_file}", flush=True)
    mParallel(n_jobs=n_jobs, verbose=30)\
        (delayed(extract_chip)\
                (source_file, lat_field, lon_field, chip.geometry, chip.identifier, tiles_folder) \
                    for _ ,chip in tiles.iterrows())