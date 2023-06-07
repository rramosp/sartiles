import argparse
from . import gunwcmds 
from . import __version__

def main():


    


    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='commands', dest='cmd')
    
    grid_parser = subparsers.add_parser('gunw', help='queries ASF to find what granules correspond to what tiles')
    grid_parser.add_argument('--tiles_file', required=True, type=str, help="geopandas dataframe containing tiles, with 'geometry' and 'identifier' columns")
    grid_parser.add_argument('--tiles_folder', required=True, type=str, help='where to store the resulting tiles')
    grid_parser.add_argument('--granules_download_folder', required=True, type=str, help='where to download the granules from ASF before tiling them')
    grid_parser.add_argument('--year', required=True, type=int, help='year to query')
    grid_parser.add_argument('--month', required=True, type=int, help='month to query')
    grid_parser.add_argument('--no_retry', default=False, action='store_true', help='if set, skipped tiles in previous runs will not be retried')
    grid_parser.add_argument('--n_jobs', default=-1, type=int, help='number of parallel jobs (defaults to -1, using all CPUs)')

    print ("-----------------------------------------------------------")
    print (f"JPL's Aria GUNW dataset extractor utility {__version__}")
    print ("-----------------------------------------------------------")
    print ()
    args = parser.parse_args()

    if args.cmd=='gunw':
        print ("-----------------------------------------------------------")
        print (f"JPL's Aria GUNW dataset extractor utility {__version__}")
        print ("-----------------------------------------------------------")
        print ()

        print ("mapping tiles to granules", flush=True)
        gunwcmds.download(
                       tiles_file               = args.tiles_file, 
                       tiles_folder             = args.tiles_folder,
                       granules_download_folder = args.granules_download_folder,
                       year                     = args.year, 
                       month                    = args.month,
                       n_jobs                   = args.n_jobs,
                       no_retry                 = args.no_retry
                      )


