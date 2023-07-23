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
import requests
import csv
from bs4 import BeautifulSoup
import configparser

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
    
    asf_baseurl='https://api.daac.asf.alaska.edu/services/search/param?'

    lastday_in_month = calendar.monthrange(year, month)[1]
    start_date = f"{year:4d}-{month:02d}-01"
    end_date   = f"{year:4d}-{month:02d}-{lastday_in_month:02d}"
    
    p = geom.wkt

    attempts = 1
    retry = True
    while retry:

        # make http request
        conf = dict(
            output = 'csv',
            processingLevel = 'GUNW_STD',
            intersectsWith = p,
            start = f'{start_date}T00:00:00UTC',
            end = f'{end_date}T23:59:59UTC'
        )

        arg_str='&'.join('%s=%s'%(k,v) for k,v in conf.items())
        argurl=asf_baseurl + arg_str
        r=requests.post(argurl)

        if r.status_code != 200:
            return f"ASFQUERY_ERROR in HTTP request {r.text}"
            
        # parse result into a dataframe
        reader = csv.DictReader(r.text.splitlines())
        rows=list(reader)
        z = pd.DataFrame(rows)

        # this signals an error condition as an HTML in the HTTP response
        if '<html>' in z.columns:                
            msg = BeautifulSoup(rows, 'lxml').find('body').get_text().strip()    
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
        p1 = i[[ 'Near Start Lon','Near Start Lat']].values.astype(float)
        p2 = i[['Far Start Lon', 'Far Start Lat']].values.astype(float)
        p3 = i[['Near End Lon', 'Near End Lat']].values.astype(float)
        p4 = i[['Far End Lon', 'Far End Lat']].values.astype(float)

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


def select_deltadays_lte_48(zz):
    return zz[zz['Days in Pair']<=48]

def get_delta_days(date_pair):
    d0 = datetime.strptime(date_pair.split("_")[0], "%Y%m%d")
    d1 = datetime.strptime(date_pair.split("_")[1], "%Y%m%d")
    delta_days = (d0-d1).days        
    return delta_days

class GUNWGranule:
    
    def __init__(self, url, cache_folder):
        self.url = url
        self.granule_name = self.url.split("/")[-1][:-3]        
        self.cache_folder = cache_folder
        self.local_file_basename = f"{self.cache_folder}/{self.granule_name}.nc"
        
        self.date_pair = self.granule_name.split("-")[6]
        self.delta_days = get_delta_days(self.date_pair)      
        self.direction = self.granule_name.split("-")[2]  
        
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
            except Exception as e:
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

        patch_data = xz.sel(longitude=slice(*lon_range), latitude=slice(*lat_range)).copy()    
        # add a datepair dimension with a single coordinate
        patch_data = patch_data.expand_dims(dim = {"datepair": [self.date_pair]}, axis=0).copy()
        xz.close()

        # add imaging geometry mean values for this patch
        lat = np.mean([maxlat, minlat])
        lon = np.mean([maxlon, minlon])

        xz = xr.open_dataset(self.local_file, 
                                    engine='netcdf4',
                                    group='science/grids/imagingGeometry')
        patch_geometry = xz.sel(longitudeMeta=lon, latitudeMeta=lat, method='nearest')
        # add a datepair dimension with a single coordinate
        patch_geometry = patch_geometry.expand_dims(dim = {"datepair": [self.date_pair]}, axis=0).copy()

        xz.close()

        

        # add radar metadata
        xz = xr.open_dataset(self.local_file, 
                                    engine='netcdf4',
                                    group='science/radarMetaData')
        
        # the matchup dimension seems to be never useds (only presents values
        # in the first coordinate)
        for v in xz.variables:
            if 'matchup' in xz[v].dims:
                xz[v] = xz[v].values[0]
        # add a datepair dimension with a single coordinate
        patch_metadata = xz.expand_dims(dim = {"datepair": [self.date_pair]}, axis=0).copy()
        
        xz.close()


        # add other metadata
        patch_extra = xr.Dataset(
            data_vars=dict(
                deltadays=np.array([self.delta_days], dtype='uint8'),
                direction=(["datepair"], [self.direction])
            ),
            coords=dict(
                datepair=patch_metadata.coords['datepair'],
            ),
            attrs=dict(description="calculated metadata."),
        )

        return {'data': patch_data, 'geom': patch_geometry, 'meta': patch_metadata, 'extra': patch_extra}


def touch(filename, content):
    with open(filename, "w") as f:
        f.write(content+"\n")

def download_granules(tile_granules, granules_download_folder, username, password):
    granules = []
    for url in tile_granules.URL:
        gw = GUNWGranule(url, cache_folder=granules_download_folder).download(username=username, password=password)
        granules.append(gw)
    return granules

def download( tiles_file, 
                    tiles_folder, 
                    granules_download_folder, 
                    year, 
                    month, 
                    mode='most-frequent',
                    username=None, 
                    password=None,
                    no_retry = False,
                    n_jobs = -1,                    
                    g = None,
                    global_asf_query_result = None
                    ):
    
    if not mode in ['most-frequent', 'lte48']:
        raise ValueError("mode must be one of 'most-frequent', 'lte48'")

    if username is None:

        cfgfile = f"{os.environ['HOME']}/.asf.cfg" 
        if os.path.isfile(cfgfile):
            config = configparser.ConfigParser()
            config.read(cfgfile)
            username = config['default']['username']
            password = config['default']['password']
            print (f"read ASF username/password from {cfgfile}")
        else:
            print (f"cfg file {cfgfile} not found, asking for credentials")
            username = input('ASF username')
            password = getpass.getpass('ASF password')        

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
                    delayed(download_job)( 
                        chip                     = chip, 
                        tiles_folder             = tiles_folder, 
                        granules_download_folder = granules_download_folder, 
                        global_asf_query_result  = global_asf_query_result,
                        mode                     = mode,
                        username                 = username, 
                        password                 = password,
                        no_retry                 = no_retry)
                     for _,chip in g.sample(len(g)).iterrows()
            ) 

def download_job( chip, 
                        tiles_folder, 
                        granules_download_folder, 
                        global_asf_query_result,
                        username, 
                        password, 
                        mode='most-frequent',
                        no_retry = False):

    z = global_asf_query_result
    retry_skipped = not no_retry

    tile = chip.geometry
    dest_file = f"{tiles_folder}/{chip.identifier}.nc"
    skipped_file = f"{tiles_folder}/{chip.identifier}.skipped"
    # if already processed, skip
    #print ("dest file", dest_file)
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

    # find in global query. this query is made by ASF using the bounding box
    # of the granule, so it might not necesarily contain the tile. 
    # The loop below checks this.
    zz = z[[i.contains(tile) for i in z.geometry]]
    
    patches = []
    
    while (len(zz)>0):

        if mode=='most-frequent':
            tile_granules = select_starttime_most_frequent_any_direction(zz)
        else:
            tile_granules = select_deltadays_lte_48(zz)

        # if not found skip it
        if tile_granules is None or len(tile_granules)==0:
            touch(skipped_file, 'NO_GRANULES_FOUND')

        # download granules
        try:
            granules = download_granules(tile_granules, granules_download_folder, username, password)
        except:
            touch(skipped_file, 'COULD_NOT_DOWNLOAD')
            return

        # retrieve  patches from all granules 
        patches = []
        boundaries = []
        urls_used = []
        datepairs_used = []
        for url in tile_granules.URL.values:
            gw = GUNWGranule(url, cache_folder=granules_download_folder)
            # in case the file falls 100% in two granules we just take the first one
            if gw.date_pair in datepairs_used:
                continue
            datepairs_used.append(gw.date_pair)
            gw.download(username=username, password=password)
            # we only consider granules that contain 100% the tile, since otherwise would require
            # collating patches from different granules with the same exact datepair, etc.
            boundaries.append(gw.get_boundary())
            if gw.get_boundary().contains(tile):
                pp = gw.get_chip(chip.identifier, chip.geometry)
                patches.append(pp)
                urls_used.append(url)
                        
        if len(patches)==0:
            # if tile is not contained in any selected granule, continue looking
            zz = zz[~zz['Granule Name'].isin(tile_granules['Granule Name'])]
        else:
            # retrict the list of granules to the ones actually used
            tile_granules = tile_granules[tile_granules.URL.isin(urls_used)]
            break 
        
    if len(patches)==0:
        touch(skipped_file, 'TILE_NOT_FULLY_CONTAINED_IN_ANY_GRANULE')
        return

    
    # set lon/lat to be the same as patch 0 
    for p in patches[1:]:
        p['data']['latitude'] = patches[0]['data'].latitude
        p['data']['longitude'] = patches[0]['data'].longitude
    
    # combine all patches
    try:
        rdata = xr.merge([p['data'] for p in patches])
        rgeom = xr.concat([p['geom'] for p in patches], dim='datepair')
        rmeta = xr.concat([p['meta'] for p in patches], dim='datepair')
        rextra = xr.concat([p['extra'] for p in patches], dim='datepair')
    except Exception as e:
        print ("\n\n\n--exception merging--")
        print (f"\nERROR ON tile {chip.identifier} ")
        print ("\n-----------------------")
        raise e

    # crs is unique
    rdata['crs'] = patches[0]['data'].crs[0]
    rgeom['crs'] = patches[0]['data'].crs[0]
    rmeta['crs'] = patches[0]['data'].crs[0]
    rextra['crs'] = patches[0]['data'].crs[0]
    rgeom['crsMeta'] = rgeom['crsMeta'].astype(np.float32)
    
    rdata.to_netcdf(dest_file, mode='w', group='/science/grids/data')
    rgeom.to_netcdf(dest_file, mode='a', group='/science/grids/imagingGeometry')
    rmeta.to_netcdf(dest_file, mode='a', group='/science/radarMetaData')
    rextra.to_netcdf(dest_file, mode='a', group='/science/extraMetaData', encoding={'deltadays': {'dtype':'uint8'}})
    
    
    rdata.close()
    rgeom.close()
    rmeta.close()
    rextra.close()
    for p in patches:
        p['data'].close()
        p['geom'].close()
        p['meta'].close()
        p['extra'].close()
    return


