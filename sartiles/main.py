import argparse
import importlib.resources as pkg_resources
import pandas as pd
import os
from . import gunw 
from . import gssic
from . import chop
from . import data
from . import __version__


tilelinks_fname = (pkg_resources.files(data) / 'Sentinel1-1_Coherence_Tiles_FileList')
tilelinks = {'file': None}
def get_tilelinks(verbose=False):
    """
    returns dataframe with the list of tile links, caching it
    """
    if tilelinks['file'] is None:

        csvfile = f'{tilelinks_fname}.csv.gz'
        pqtfile = f'/tmp/Sentinel1-1_Coherence_Tiles_FileList.parquet'

        if os.path.isfile(pqtfile):
            if verbose:
                print ("reading links file from parquet")
            tilelinks['file'] = pd.read_parquet(pqtfile)
        else:
            if verbose:
                print ("reading links file from csv")
            tilelinks['file'] = pd.read_csv(csvfile, index_col=0)
            if verbose:
                print ("saving tile links to parquet for faster retrieval")
            tilelinks['file'].to_parquet(pqtfile)

    return tilelinks['file']

def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='commands', dest='cmd')
    
    grid_parser = subparsers.add_parser('gunw', help='downloads tiles for JPLs GUNW data collection, from ASF')
    grid_parser.add_argument('--tiles_file', required=True, type=str, help="geopandas dataframe containing tiles, with 'geometry' and 'identifier' columns")
    grid_parser.add_argument('--tiles_folder', required=True, type=str, help='where to store the resulting tiles')
    grid_parser.add_argument('--granules_download_folder', required=True, type=str, help='where to download the granules from ASF before tiling them')
    grid_parser.add_argument('--year', required=True, type=int, help='year to query')
    grid_parser.add_argument('--month', required=True, type=int, help='month to query')
    grid_parser.add_argument('--no_retry', default=False, action='store_true', help='if set, skipped tiles in previous runs will not be retried')
    grid_parser.add_argument('--n_jobs', default=-1, type=int, help='number of parallel jobs (defaults to -1, using all CPUs)')


    grid_parser = subparsers.add_parser('gssic', help='downloads tiles for Global Seasonal Coherence collection, from ASF')
    grid_parser.add_argument('--tiles_file', required=True, type=str, help="geopandas dataframe containing tiles, with 'geometry' and 'identifier' columns")
    grid_parser.add_argument('--tiles_folder', required=True, type=str, help='where to store the resulting tiles')
    grid_parser.add_argument('--granules_download_folder', required=True, type=str, help='where to download the granules from ASF before tiling them')
    grid_parser.add_argument('--no_retry', default=False, action='store_true', help='if set, skipped tiles in previous runs will not be retried')
    grid_parser.add_argument('--n_jobs', default=-1, type=int, help='number of parallel jobs (defaults to -1, using all CPUs)')

    grid_parser = subparsers.add_parser('chop', help='chips any xarray readable (netcfd, tif) file into tiles')
    grid_parser.add_argument('--tiles_file', required=True, type=str, help="geopandas dataframe containing tiles, with 'geometry' and 'identifier' columns")
    grid_parser.add_argument('--tiles_folder', required=True, type=str, help='where to store the resulting tiles')
    grid_parser.add_argument('--source_file', required=True, type=str, help='the xarray readable file to chop')
    grid_parser.add_argument('--n_jobs', default=-1, type=int, help='number of parallel jobs (defaults to -1, using all CPUs)')



    print ("-----------------------------------------------------------")
    print (f"SAR datasets download and tiling utility {__version__}")
    print ("-----------------------------------------------------------")

    args = parser.parse_args()

    if args.cmd=='gunw':
        print (f"Target: JPL's Aria GUNW data collection ")
        print ("-----------------------------------------------------------")
        print ()

        print ("mapping tiles to granules", flush=True)
        gunw.download(
                       tiles_file               = args.tiles_file, 
                       tiles_folder             = args.tiles_folder,
                       granules_download_folder = args.granules_download_folder,
                       year                     = args.year, 
                       month                    = args.month,
                       n_jobs                   = args.n_jobs,
                       no_retry                 = args.no_retry
                      )

    elif args.cmd=='gssic':
        print (f"Target: Global Seasonal Coherence data collection")
        print ("-----------------------------------------------------------")
        print ()
        # preload tiles list
        get_tilelinks(verbose=True)        
        gssic.download(tiles_file               = args.tiles_file, 
                       tiles_folder             = args.tiles_folder, 
                       granules_download_folder = args.granules_download_folder, 
                       n_jobs                   = args.n_jobs,
                       no_retry                 = args.no_retry,
                       get_tilelinks_fn         = get_tilelinks
        )        

    elif args.cmd=='chop':
        print (f"Chopping file")
        print ("-----------------------------------------------------------")
        print ()

        chop.chop(tiles_file   = args.tiles_file,
                  tiles_folder = args.tiles_folder,
                  source_file  = args.source_file,
                  n_jobs       = args.n_jobs)        

        