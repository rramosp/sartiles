from geetiles import utils
import shapely as sh
import getpass
from rlxutils import Command
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

def gethash(s):
    """
    returns and md5 hashcode for a string
    """
    k = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k

def query_asfgunw(geom, year, month, return_only_most_frequent_enddate=True):
    
    """
    queries asf gunw collection for all the pairs whose end date is within year/month, containing the given geometry

    geom: a shapely geometry in WSG84 lon lat degrees
    year, month: the year/month of the pairs end date
    return_only_most_frequent_enddate: if True it only returns the pairs of the latest most frequent end date
    
    retuns: a geopandas dataframe
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

    with open("/tmp/cfg", "w") as f:
        f.write(s)


    # query ASF through a command
    currentdir = os.getcwd()
    basedir = f"{os.environ['HOME']}/data"

    cmd = Command(f'python {currentdir}/lib/sentinel_query_download/sentinel_query_download.py /tmp/cfg ', 
                  cwd = f'{basedir}/tmp')
    cmd.run().wait()
    s = cmd.stdout()
    query_log = re.search(r'asf_query_(.*)\.csv', s)
    if query_log is None:
        return None
    
    query_log = query_log.group(0)
    qfile = f"{basedir}/tmp/{query_log}"
    z = pd.read_csv(qfile)
    
    # in case no results
    if len(z)==0:
        return None
    
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
    
    if return_only_most_frequent_enddate:
        # gets acquisitions in the latest same start date with most date pairs
        ztmp = z.groupby('Start Time').size()
        start_time_most_frequent = ztmp[ztmp==ztmp.max()].sort_index(ascending=False).index[0]
        z = z[z['Start Time']==start_time_most_frequent]

        
    return z


def tiles2granules(tiles_file, year, month):

    base_filename = os.path.splitext(tiles_file)[0]

    granule_list = None
    granules_for_chip = {}

    granule_list_file = f"{base_filename}_{year}-{month:02d}_granuledefs.geojson"
    granules_for_chip_file = f"{base_filename}_{year}-{month:02d}_granulemap.pkl"


    print ("reading tiles", flush=True)
    g = gpd.read_file(tiles_file)

    # load from previous run if available

    if os.path.isfile(granule_list_file):
        granule_list = gpd.read_file(granule_list_file)
        
    if os.path.isfile(granules_for_chip_file):
        with open(granules_for_chip_file, 'rb') as f:
            granules_for_chip = pickle.load(f)
            

    print ("mapping tiles to GUNW granules", flush=True)
    # iterate geometries
    for i,(_, chip) in pbar(enumerate(g.iterrows()), max_value=len(g)):
        geom = chip.geometry
        
        if chip.identifier in granules_for_chip.keys():
            continue
        
        if granule_list is None:
            granule_list = query_asfgunw(geom, year, month)
            if granule_list is None:
                granules_for_chip[chip.identifier] = []
                continue
            granules_for_chip[chip.identifier] = list(granule_list['hash'].values)

        # if any granule interects the chip without fully containing it
        # then, ignore the chip since it would require to stick parts
        # of different pairs, probably with different dates, etc. 
        contains = [i.contains(geom) for i in granule_list.geometry]
        intersects = [i.intersects(geom) for i in granule_list.geometry]

        if sum(intersects)>sum(contains):
            granules_for_chip[chip.identifier] = []
            continue
            
        # if no current granule contains the chip, then make asf query
        if sum(contains)==0:
            chip_granules = query_asfgunw(geom, year, month)
            if chip_granules is None:
                granules_for_chip[chip.identifier] = []
                continue
            chip_granules.index = range(len(granule_list), len(granule_list)+len(chip_granules))

            granule_list = pd.concat([granule_list, chip_granules])
            granules_for_chip[chip.identifier] = list(chip_granules['hash'].values)

        # if chip already contained in some granule, add the hash codes of the granules
        # containing it
        else:
            granules_for_chip[chip.identifier] = list(granule_list['hash'][contains].values)
            

        if i%100==0  and granule_list is not None:
            granule_list.to_file(granule_list_file, driver='GeoJSON')
            with open(granules_for_chip_file, 'wb') as f:
                pickle.dump(granules_for_chip, f)
