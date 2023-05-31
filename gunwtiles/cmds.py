from geetiles import utils
import shapely as sh
import getpass
from rlxutils import Command, mParallel
from joblib import delayed
import os
import re
import pandas as pd
import geopandas as gpd
from pyproj import CRS
import numpy as np
import calendar
import hashlib
from progressbar import progressbar as pbar
import pickle
from datetime import datetime
import xarray as xr
from skimage.transform import resize
from pathlib import Path
from glob import glob
from bs4 import BeautifulSoup
from time import sleep

def gethash(s):
    """
    returns and md5 hashcode for a string
    """
    k = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k

def query_asfgunw(geom, year, month, username, password, debug=False):
    
    """
    queries asf gunw collection for all the pairs whose end date is within year/month, containing the given geometry

    geom: a shapely geometry in WSG84 lon lat degrees
    year, month: the year/month of the pairs end date
    
    retuns: a geopandas dataframe if sucessful
            a string with error message otherwise
    """
    
    lastday_in_month = calendar.monthrange(year, month)[1]
    start_date = f"{year:4d}-{month:02d}-01"
    end_date   = f"{year:4d}-{month:02d}-{lastday_in_month:02d}"
    
    p = geom.wkt

    s = f"""
    [api_search]
    output = csv
    processingLevel = GUNW_STD
    intersectsWith = {p}
    start = {start_date}T00:00:00UTC
    end = {end_date}T00:00:00UTC

    [download]
    download_site = both
    nproc = 2

    [asf_download]
    http-user = {username}
    http-password = {password}
    """

    basedir = f"/tmp"
    cfg_file = f"{basedir}/cfg_{np.random.randint(1000000000):11d}"
    with open(cfg_file, "w") as f:
        f.write(s)

    script = str(os.path.dirname(__file__))+"/sentinel_query_download/sentinel_query_download.py"    

    attempts = 1
    retry = True
    while retry:
        # query ASF through a command
        cmd = Command(f'python {script} {cfg_file} ', 
                    cwd = f'{basedir}')
        if debug:
            print ("cmd is", cmd.cmd)
        cmd.run().wait()
        s = cmd.stdout()

        # parse the output to retrieve the filename with the results
        query_log = re.search(r'asf_query_(.*)\.csv', s)
        os.remove(cfg_file)
        if debug:
            print ("cmd status", cmd.exitcode())
            print ("\n\ncmd stdout\n", cmd.stdout())
            print ("\n\ncmd stderr\n", cmd.stderr())

        if query_log is None:
            print ("cmd" cmd.cmd)
            print ("cmd status", cmd.exitcode())
            print ("\n\ncmd stdout\n", cmd.stdout())
            print ("\n\ncmd stderr\n", cmd.stderr())
            return "ASFQUERY_NO_LOG"
        
        # load the results
        query_log = query_log.group(0)
        qfile = f"{basedir}/{query_log}"
        if debug:
            print ("query result", qfile)
        z = pd.read_csv(qfile)

        # this signals an error condition as an HTML in the HTTP response
        if '<html>' in z.columns:
            with open(qfile) as f:
                html = f.read()
                
            from bs4 import BeautifulSoup
            msg = BeautifulSoup(html, 'lxml').find('body').get_text().strip()    
            if "Time-out" in msg and attempts<4:
                print (f"ASF time out at attempt {attempts}, sleeping 10s")
                sleep(10)
                attempts += 1
                continue

            return f'ASFQUERY_ERROR {msg}'

        retry = False

    # in case no results
    if len(z)==0:
        return "ASFQUERY_NO_RESULTS"
    
    # in case other output format
    if not 'Granule Name' in z.columns:
        return "ASFQUERY_INCORRECT_FORMAT"
    
    # add date pair info
    z['Date Pair'] = [i[6] for i in z['Granule Name'].str.split("-")]
    
    ndays = []
    for dp in z['Date Pair']:
        d0 = datetime.strptime(dp.split("_")[0], "%Y%m%d")
        d1 = datetime.strptime(dp.split("_")[1], "%Y%m%d")
        ndays.append((d0-d1).days)
    z['Days in Pair'] = ndays
        

    # add hash code for each granule
    z['hash'] = [gethash(i) for i in z['Granule Name']]      
        
    geoms = []
    for _,i in z.iterrows():
        p1 = i[[ 'Near Start Lon','Near Start Lat']].values
        p2 = i[['Far Start Lon', 'Far Start Lat']].values
        p3 = i[['Near End Lon', 'Near End Lat']].values
        p4 = i[['Far End Lon', 'Far End Lat']].values

        p = sh.geometry.Polygon([p1, p3, p4, p2])
        geoms.append(p)

    z['geometry'] = geoms
    z = gpd.GeoDataFrame(z, crs=CRS.from_epsg(4326))    

    return z

def select_starttime_most_frequent(zz, direction):
    if not direction in ['asc', 'desc']:
        raise ValueError("invalid direction specficiation")
        
    if direction=='asc':
        zz = zz[zz['Ascending or Descending?']=='ascending']
    else:
        zz = zz[zz['Ascending or Descending?']=='descending']
        
    if len(zz)==0:
        return zz
        
    ztmp = zz.groupby('Start Time').size()
    start_time_most_frequent = ztmp[ztmp==ztmp.max()].sort_index(ascending=False).index[0]
    return zz[zz['Start Time']==start_time_most_frequent]


def select_starttime_most_frequent_any_direction(zz):
    tile_granules = select_starttime_most_frequent(zz, direction='asc')
    if len(tile_granules)==0:
        tile_granules = select_starttime_most_frequent(zz, direction='desc')
    return tile_granules


class GUNWGranule:
    
    def __init__(self, url, cache_folder):
        self.url = url
        self.granule_name = self.url.split("/")[-1][:-3]        
        self.cache_folder = cache_folder
        self.local_file_basename = f"{self.cache_folder}/{self.granule_name}.nc"
        
        self.date_pair = self.granule_name.split("-")[6]
        d0 = datetime.strptime(self.date_pair.split("_")[0], "%Y%m%d")
        d1 = datetime.strptime(self.date_pair.split("_")[1], "%Y%m%d")
        self.delta_days = (d0-d1).days        
        
        
    def download(self, username, password):
                
        # tries to find any file with this basename. this is done so that parallel processes
        # can attempt to download the same file independently and posterior calls will try
        # to find anyone which has already been downloaded.
        self.local_file = None
        for filename in glob(f"{self.local_file_basename}*"):
            try:
                xz = xr.open_dataset(filename, 
                        engine='netcdf4',
                        group='science/grids/data')
                xz.close()
                self.local_file = filename
                break
            except:
                continue
                
        if self.local_file is None:
            self.local_file = f"{self.local_file_basename}_{np.random.randint(100000000):09d}"

        else:
            return self

        print ("downloading", self.url, flush=True)
        cmd_string = f"wget --output-document={self.local_file} --http-user={username} --http-password={password} {self.url}"
        cmd = Command(cmd_string, cwd=self.cache_folder)
        cmd.run().wait(raise_exception_on_error=True)
        if cmd.exitcode()!=0:
            raise ValueError(f"error downloading. cmd is {cmd_string}\n\n exit code {cmd.exitcode()} \n\nstderr\n{cmd.stderr()} \n\nstdout\n{cmd.stdout()}")

        if not os.path.isfile(self.local_file):
            raise ValueError(f"{self.granule_name} not downloaded")
            
        return self
    
    def get_boundary(self):
        """
        returns the granule boundary as a shapefile object
        """
        z = xr.open_dataset(self.local_file, engine='netcdf4',)
        bb = z.variables['productBoundingBox'].values[0].decode()
        z.close()
        return sh.wkt.loads(bb)
    
    
    def get_chip(self, chip_identifier, chip_geometry): 
        
        is_ascending = lambda x: x[-1]>x[0]        
        
        xz = xr.open_dataset(self.local_file, 
                            engine='netcdf4',
                            group='science/grids/data')
        
        tile = chip_geometry
        tile_coords = np.r_[list(tile.boundary.coords)]
        minlon, minlat = np.min(tile_coords, axis=0)
        maxlon, maxlat = np.max(tile_coords, axis=0)        
        
        if not len(np.unique(tile_coords[:,0]))==2 or not len(np.unique(tile_coords[:,1]))==2:
            raise ValueError(f'tile {chip_identifier} must have bounding box coordinates alined with latitude and longitude')

        if is_ascending(xz.coords['latitude'].values):
            lat_range = (minlat, maxlat)
        else:
            lat_range = (maxlat, minlat)

        if is_ascending(xz.coords['longitude'].values):
            lon_range = (minlon, maxlon)
        else:
            lon_range = (maxlon, minlon)

        patch = xz.sel(longitude=slice(*lon_range), latitude=slice(*lat_range))        
        xz.close()
        return patch


def touch(filename, content):
    with open(filename, "w") as f:
        f.write(content+"\n")

def download_granules(tile_granules, granules_download_folder, username, password):
    granules = []
    for url in tile_granules.URL:
        gw = GUNWGranule(url, cache_folder=granules_download_folder).download(username=username, password=password)
        granules.append(gw)
    return granules

def tiles2granules( tiles_file, 
                    tiles_folder, 
                    granules_download_folder, 
                    year, 
                    month, 
                    username=None, 
                    password=None,
                    no_retry = False,
                    n_jobs = -1,                    
                    g = None,
                    global_asf_query_result = None
                    ):
    if username is None:
        username =   input("ASF username: ")
    
    if password is None:
        password = getpass.getpass("ASF password: ")

    if g is None:
        print ("reading tiles file", flush=True)
        g = gpd.read_file(tiles_file)


    if global_asf_query_result is None:
        print ("building region geometry", flush=True)

        # we sample g since we want the bounding box and this is a good approximation
        ch = utils.concave_hull(list(g.sample(20000).geometry), use_pbar=True)            

        print ("making query to ASF GUNW", flush=True)
        global_asf_query_result = query_asfgunw(ch.simplify(tolerance=.5), 
                                                year=year, 
                                                month=month, 
                                                username=username, 
                                                password=password )
        
        if isinstance(global_asf_query_result, str):
            # this signals an error condition
            print (f"error making global query", global_asf_query_result)
            return


    print (f"downloading {len(g)} tiles", flush=True)
    
    mParallel(n_jobs=n_jobs, verbose=30)(
                    delayed(tiles2granules_job)( 
                        chip                     = chip, 
                        tiles_folder             = tiles_folder, 
                        granules_download_folder = granules_download_folder, 
                        global_asf_query_result  = global_asf_query_result,
                        year                     = year, 
                        month                    = month,
                        username                 = username, 
                        password                 = password,
                        no_retry                 = no_retry)
                     for _,chip in g.sample(len(g)).iterrows()
            ) 

    

def tiles2granules_job( chip, 
                        tiles_folder, 
                        granules_download_folder, 
                        global_asf_query_result,
                        year, 
                        month,
                        username, 
                        password, 
                        no_retry = False):

    z = global_asf_query_result
    retry_skipped = not no_retry

    tile = chip.geometry
    dest_file = f"{tiles_folder}/{chip.identifier}.nc"
    skipped_file = f"{tiles_folder}/{chip.identifier}.skipped"
        
    # if already processed, skip
    if os.path.isfile(dest_file):
        return
        
    # if attempted but failed, retry if requested in case
    # of transitory errors (could not connect, etc.)
    if retry_skipped and os.path.isfile(skipped_file):
        with open(skipped_file) as f:
            errorcode = f.read()
            if errorcode.strip() in ['COULD_NOT_DOWNLOAD']:
                pass   # method will continue and tile fownload will be retried
            else:
                return # dont do anything, no retry

    # find in global query
    zz = z[[i.contains(tile) for i in z.geometry]]
    tile_granules = select_starttime_most_frequent_any_direction(zz)
    
    # if not found skip it
    if tile_granules is None or len(tile_granules)==0:
        touch(skipped_file, 'NO_GRANULES_FOUND')
        return
    
    # download granules
    try:
        granules = download_granules(tile_granules, granules_download_folder, username, password)
    except:
        touch(skipped_file, 'COULD_NOT_DOWNLOAD')
        return


    # check the chip is actualy contained, as the geometries of the global query
    # are the bounding boxes of the granules and not the granules themselves.
    # caching in GUNWGranule.download will allow to reuse previous downloads in new chips
    if not np.alltrue([ gw.get_boundary().contains(tile) for gw in granules]):
        print (f"doing ASF query for tile {chip.identifier}")
        tile_granules = query_asfgunw(tile, year=year, month=month, username=username, password=password)
        if isinstance(tile_granules, str):
            # this signals an error condition
            touch(skipped_file, tile_granules)
            return

        if tile_granules is not None and len(tile_granules)>0:
            tile_granules = select_starttime_most_frequent_any_direction(tile_granules)
            # download again
            try:
                granules = download_granules(tile_granules, granules_download_folder, username, password)
            except Exception as e:
                touch(skipped_file, 'COULD_NOT_DOWNLOAD')
                return

        else:
            touch(skipped_file, 'TILE_NOT_CONTAINED_IN_GLOBAL_QUERY')
            return

            
    # retrieve  patches from all granules 
    patches = []
    skip_tile = False
    for url in tile_granules.URL.values:
        gw = GUNWGranule(url, cache_folder=granules_download_folder)
        gw.download(username=username, password=password)
        # we check this again to discard tiles not 100% contained within the granule, since this would require
        # collating patches from different granules with the same exact datepair, etc.
        if not gw.get_boundary().contains(tile):
            skip_tile = True
            break

        pp = gw.get_chip(chip.identifier, chip.geometry)
        patches.append(pp)

    if skip_tile:
        touch(skipped_file, 'TILE_NOT_CONTAINED_IN_TILE_SPECIFIC_QUERY')
        return


    # collate them into a single xarray and save it as NetCDF
    # resizing them if necessary
    p = patches[0]
    vars2d = [v for v in p.variables if len(p[v].coords)==2]    
    patch_shape = patches[0][vars2d[0]].values.shape
    vars2d_data = {v: np.r_[[resize(patch[v], patch_shape, preserve_range=True) for patch in patches]] for v in vars2d}

    # if there is any nan, skip
    if sum([np.sum(np.isnan(v)) for k,v in vars2d_data.items()])>0:
        touch(skipped_file, 'TILE_WITH_NANS')
        return


    for v in vars2d:
        if len(np.unique([" ".join(p[v].dims) for p in patches])) != 1:
            raise ValueError(f"lonlat coords in variable {v} are not in the same order in all patches")

            
    r = patches[0].copy()
    r = r.expand_dims(dim = {"datepair": list(tile_granules['Date Pair'].values)}, axis=0).copy()
    for v in vars2d:
        r[v] = (('datepair', *p[v].dims), vars2d_data[v] )
    
    r.to_netcdf(dest_file)
    r.close()

