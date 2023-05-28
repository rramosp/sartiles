import argparse
from .cmds import *
from . import __version__

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='commands', dest='cmd')
    
    grid_parser = subparsers.add_parser('tiles2granules', help='queries ASF to find what granules correspond to what tiles')
    grid_parser.add_argument('--tiles_file', required=True, type=str, help="geopandas dataframe containing tiles, with 'geometry' and 'identifier' columns")
    grid_parser.add_argument('--year', required=True, type=int, help='year to query')
    grid_parser.add_argument('--month', required=True, type=int, help='month to query')



    print ("-----------------------------------------------------------")
    print (f"JPL's Aria GUNW dataset extractor utility {__version__}")
    print ("-----------------------------------------------------------")
    print ()
    args = parser.parse_args()

    if args.cmd=='tiles2granules':

        print ("mapping tiles to granules", flush=True)
        tiles2granules(tiles_file = args.tiles_file, 
                       year       = args.year, 
                       month      = args.month)


