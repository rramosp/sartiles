import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
from glob import glob
import importlib.resources as pkg_resources
from rlxutils import Command, mParallel
from joblib import delayed
import os
from rasterio.transform import Affine
from . import data
import matplotlib.pyplot as plt
from rlxutils import subplots 
import shapely as sh
from rioxarray.merge import merge_datasets
import configparser
import getpass
import geopandas as gpd
from geetiles import utils

tilelinks_fname = (pkg_resources.files(data) / 'Sentinel1-1_Coherence_Tiles_FileList.csv.gz')
tilelinks = {'file': None}

def check_xy_coords_equal(patch_list):
    csetx = [p.coords['x'].values for p in patch_list]
    csety = [p.coords['y'].values for p in patch_list]

    # check all coordinates have the same number of elements
    lens = np.r_[[len(c) for c in csetx]]
    if not np.all(lens[0]==lens):
        raise ValueError("patches with different coordinate x length")

    lens = np.r_[[len(c) for c in csety]]
    if not np.all(lens[0]==lens):
        raise ValueError("patches with different coordinate y length")

    # check all coordinates are equal
    if not np.allclose(np.r_[csetx].std(axis=0), 0):
        raise ValueError("patches with different x coordinate")

    if not np.allclose(np.r_[csety].std(axis=0), 0):
        raise ValueError("patches with different y coordinate")

    return True
    

def get_tilelinks():
    """
    returns dataframe with the list of tile links, caching it
    """
    if tilelinks['file'] is None:
        print ("reading links file")
        tilelinks['file'] = pd.read_csv(tilelinks_fname, index_col=0)

    return tilelinks['file']


def lonlat2tileid(lon, lat):
    lon_prefix = 'W' if lon<0 else 'E'
    lat_prefix = 'S' if lat<0 else 'N'

    if lon<0:
        lon_prefix = 'W'
        lon_tile   = np.ceil(np.abs(lon)).astype(int)
    else:
        lon_prefix = 'E'
        lon_tile   = np.floor(np.abs(lon)).astype(int)

    if lat<0:
        lat_prefix = 'S'
        lat_tile   = np.floor(np.abs(lat)).astype(int)
        if lat_tile==0:
            lat_prefix = 'N'
    else:
        lat_prefix = 'N'
        lat_tile   = np.ceil(np.abs(lat)).astype(int)


    return f'{lat_prefix}{lat_tile:02d}{lon_prefix}{lon_tile:03d}'


def plot_chip(c):
    for ax,g in subplots(c.feature.values, usizex=4, usizey=2.5):
        plt.imshow(c.geometry.sel({'feature': g}).values)   
        plt.title(g)
        plt.colorbar()

    for p in c.polarimetry.values:
        for ax,s in subplots(c.season.values, usizex=5):
            vmax = 8000 if p=='vv' else 4000
            plt.imshow(c.amplitude.sel({'polarimetry': p, 'season': s}).values, vmin=0, vmax=vmax)   
            plt.title(f'amplitude {p} {s}')
            plt.colorbar()


    for d in c.deltadays.values:
        for ax,s in subplots(c.season.values, usizex=5):
            plt.imshow(c.coherence.sel({'deltadays': d, 'season': s}).values, vmin=0, vmax=80)   
            plt.title(f'coh {d} days {s}')
            plt.colorbar()

class GSSICTile:
    
    geometry_component_names = [
                       'inc', 'lsmap'
    ]
        
    coh_component_names    = ['vv_COH12', 'vv_COH24', 'vv_COH36', 'vv_COH48']    
    pol_components_names   = ['vv_AMP', 'vh_AMP' ]
    fit_component_names    = ['vv_rho', 'vv_tau', 'vv_rmse']    
    season_component_names = pol_components_names + coh_component_names + fit_component_names

    season_names = [ 'summer', 'fall', 'winter', 'spring' ]
    
    def __init__(self, lon, lat, cache_folder):
        self.lon = lon
        self.lat = lat
        alllinks         = get_tilelinks()
        self.tileid      = lonlat2tileid(lon, lat)
        self.tilelinks   = alllinks[alllinks.TILE==self.tileid]
        self.local_files = {}
        self.random_id   = f'{np.random.randint(100000000):09d}'
        self.cache_folder      = cache_folder
        self.cache_tile_folder = f'{self.cache_folder}/{self.tileid}'
        self.reset_cache_file_list()
        
    def clone(self):
        r = self.__class__(self.lon, self.lat, self.cache_folder)
        r.cache_files = self.cache_files.copy()
        r.local_files = self.local_files.copy()
        return r

    def reset_cache_file_list(self):
        self.cache_files = {}
        return

    def get_bounds(self):
        xz = xr.open_dataset(self.cache_files['inc'], engine='rasterio')    
        lon1, lat1, lon2, lat2 = xz.rio.bounds()
        r = sh.geometry.Polygon([[lon1, lat1], [lon1, lat2], [lon2,lat2], [lon2, lat1]])
        xz.close()
        return r    

    def get_local_filename(self, component, season = None, idx=0):
        return self.get_local_basename(component, season, idx) + "."+self.random_id

    def get_full_component(self, component, season=None, idx=0):
        """
        builds a string with the component and season
        idx only used for component 'inc' or 'lsmap'
        """
        if component in self.geometry_component_names:
            if season is not None:
                raise ValueError("cannot specify season for components 'inc' or 'lsmap'")                
            if idx is not None:
                return f'{component}-{idx:02d}'
            else:
                return component
        elif component in self.season_component_names:
            if not season in self.season_names:
                raise ValueError(f"invalid season '{season}'. valid are {self.season_names}")
            return f"{season}_{component}"
        else:
            raise ValueError(f"invalid component {component}")
    
    def get_local_basename(self, component, season = None, idx=0):
        return f'{self.cache_tile_folder}/{self.tileid}_{self.get_full_component(component, season, idx)}.tif'
        
    def read_component(self, component, season=None):
        """
        reads component files once they have been downloaded.
        if we have a list of files, this means that the chip overlaps
        various tiles, so we load and merge them all
        """
        c = self.get_full_component(component, season)

        files_to_read = []
        for c in self.cache_files.keys():
            if component not in c:
                continue
            if isinstance(self.cache_files[c], list):
                files_to_read += self.cache_files[c]
            else:
                files_to_read.append(self.cache_files[c])

        xz_set = [xr.open_dataset(f, engine='rasterio') for f in files_to_read]
        xz = merge_datasets(xz_set)
        for s in xz_set:
            s.close()

        return xz


    def get_component_season_patch(self, chip_identifier, chip_geometry, component, season = None):
        """
        extracts a geometry from a component/season
        """
        is_ascending = lambda x: x[-1]>x[0]        
        
        coords = np.r_[chip_geometry.boundary.coords]
        minlon, minlat = coords.min(axis=0)
        maxlon, maxlat = coords.max(axis=0)

        c = self.get_full_component(component, season)
        filename = self.cache_files[c]
        
        if filename is None:
            raise ValueError(f"no file found for {c} on chip {chip_identifier}")
            
        xz = self.read_component(component, season)
        
        if is_ascending(xz.coords['y'].values):
            lat_range = (minlat, maxlat)
        else:
            lat_range = (maxlat, minlat)

        if is_ascending(xz.coords['x'].values):
            lon_range = (minlon, maxlon)
        else:
            lon_range = (maxlon, minlon)

            
        patch_data = xz.sel(x=slice(*lon_range), y=slice(*lat_range)).copy()    

        xz.close()
        return patch_data

    def download(self, username, password):
        self.reset_cache_file_list()

        for c in self.geometry_component_names:
            self.download_component(username, password, c)
            
        for s in self.season_names:
            for c in self.season_component_names:
                self.download_component(username, password, c, s)
        
        
    def download_component(self, username, password, component, season=None ):        

        # restrict to links for the tile and components
        full_component = self.get_full_component(component, season, idx=None)
        tlinks = self.tilelinks[self.tilelinks.FILENAME.str.contains(full_component)]

        if len(tlinks)==0:
            raise ValueError(f"component {full_component} not found")
        
        for idx,url in enumerate(tlinks.LINK.values): 
            local_basename = self.get_local_basename(component, season, idx=idx)
            full_component = self.get_full_component(component, season, idx=idx)

            # tries to find any file with this basename. this is done so that parallel processes
            # can attempt to download the same file independently and posterior calls will try
            # to find anyone which has already been downloaded.
            for filename in glob(f"{local_basename}*"):
                try:
                    xz = rxr.open_rasterio(filename)
                    xz.close()
                    self.cache_files[full_component] = filename
                    break
                except Exception as e:
                    continue

            if not full_component in self.cache_files.keys() or self.cache_files[full_component] is None:
                self.cache_files[full_component] = f"{local_basename}_{self.random_id}"
            else:
                continue
            
            # if not downloaded, download it.
            os.makedirs(self.cache_tile_folder, exist_ok=True )
            
            print ("downloading", url, flush=True)
            cmd_string = f"wget --output-document={self.cache_files[full_component]} --http-user={username} --http-password={password} {url}"
            cmd = Command(cmd_string, cwd=self.cache_tile_folder)
            cmd.run().wait(raise_exception_on_error=True)
            if cmd.exitcode()!=0:
                raise ValueError(f"error downloading. cmd is {cmd_string}\n\n exit code {cmd.exitcode()} \n\nstderr\n{cmd.stderr()} \n\nstdout\n{cmd.stdout()}")

            if not os.path.isfile(self.cache_files[full_component]):
                raise ValueError(f"{self.tileid} {component} {full_component} not downloaded")
                        
        return self
                    

    def get_chip(self, chip_identifier, chip_geometry):

        # read all chip geometry in all files
        ppol = [[self.get_component_season_patch(chip_identifier, chip_geometry, c, s) \
                    for c in self.pol_components_names] for s in self.season_names] 

        pcoh = [[self.get_component_season_patch(chip_identifier, chip_geometry, c, s) \
                    for c in self.coh_component_names] for s in self.season_names] 


        pfit = [[self.get_component_season_patch(chip_identifier, chip_geometry, c, s) \
                    for c in self.fit_component_names] for s in self.season_names] 

        pgeom = [self.get_component_season_patch(chip_identifier, chip_geometry, c) \
                    for c in self.geometry_component_names]


        check_xy_coords_equal([p for pp in ppol for p in pp] +\
                              [p for pp in pcoh for p in pp] +\
                              [p for pp in pfit for p in pp] +\
                              pgeom)

        # get data arrays
        vpol  = np.r_[[[p['band_data'].values[0]  for p in pp] for pp in ppol]]
        vcoh  = np.r_[[[p['band_data'].values[0]  for p in pp] for pp in pcoh]]
        vfit  = np.r_[[[p['band_data'].values[0]  for p in pp] for pp in pfit]]
        vgeom = np.r_[[p['band_data'].values[0]  for p in pgeom]]


        # --------------------------
        # build rio xarray

        # compute dimenstions and resolution
        lons = pgeom[0].x.values
        lats = pgeom[0].y.values
        dimx = len(lons)
        dimy = len(lats)
        xres = (lons.max() - lons.min()) / dimx
        yres = (lats.max() - lats.min()) / dimy

        # build coordinates
        pols = [i[:2] for i in self.pol_components_names] 
        deltadays = [int(i[-2:]) for i in self.coh_component_names] 
        features = self.geometry_component_names
        fitparams = [i[3:] for i in self.fit_component_names]
        seasons = self.season_names


        # define raster affine transform
        transform = Affine.scale(xres, yres) * Affine.translation(lons.min() - xres / 2, lons.max() + yres / 2)
        transform = Affine.scale(xres, yres) * Affine.translation(lons.min() , lats.max() )
        transform

        # put everything together
        ds = xr.Dataset(
                coords = {'deltadays': deltadays, 
                        'polarimetry':pols, 
                        'param': fitparams,
                        'feature': features,
                        'season': seasons,
                        'x': lons, 
                        'y': lats},
                data_vars = {
                    'amplitude': (['season', 'polarimetry', 'y', 'x'], vpol),
                    'coherence': (['season', 'deltadays', 'y', 'x'], vcoh),
                    'decaymodel':(['season', 'param', 'y', 'x'], vfit),
                    'geometry':  (['feature', 'y', 'x'], vgeom)
                }
            )

        ds.rio.write_transform(transform, inplace=True)\
        .rio.write_crs('4326',inplace=True,)\
        .rio.write_coordinate_system(inplace=True)

        return ds
        
class GSSIC_Chip_Builder:
    """
    A class to manage chip creation from GSSIC tiles and storing. Takes into account
    chips overlapping across different GSSIC tiles.
    """
    
    def __init__(self, chip_identifier, chip_geometry, cache_folder):
        
        self.chip_identifier = chip_identifier
        self.chip_geometry = chip_geometry
        self.cache_folder = cache_folder
        
        coords = np.r_[chip_geometry.boundary.coords]
        coords = pd.DataFrame(coords, columns=['x', 'y'])
        coords['xr'] = np.floor(coords['x']).astype(int).astype(str)
        coords['yr'] = np.floor(coords['y']).astype(int).astype(str)
        coords['xy'] = coords.xr + coords.yr
        gtile_coords = coords.groupby('xy')[['x', 'y']].mean()
        self.gtiles = [GSSICTile(lon, lat, cache_folder=self.cache_folder) for lon, lat in gtile_coords.values]

        
    def download(self, username, password):
        for gtile in self.gtiles:
            gtile.download(username, password)

        """
        uses a copy of the first chip as placeholder to gather all files for each component
        """
        self.gmain = self.gtiles[0].clone()
        
        if len(self.gtiles)>1:
            cf = {}
            for gt in self.gtiles:
                for k,v in gt.cache_files.items():
                    if not k in cf:
                        cf[k] = []
                    cf[k].append(v)


            self.gmain.cache_files = cf        
            
    def get_chip(self):
        """
        builds chip by calling get_chip on each tile. if there are several tiles
        relies on the mechanism within GSSICTile.get_chip to deal with lists of 
        files for each component.
        """
            
        return self.gmain.get_chip(self.chip_identifier, self.chip_geometry)
        
    def read_component(self, component, season=None):
        return self.gmain.read_component(component, season)


def touch(filename, content):
    with open(filename, "w") as f:
        f.write(content+"\n")


def download(       tiles_file, 
                    tiles_folder, 
                    granules_download_folder, 
                    username=None, 
                    password=None,
                    no_retry = False,
                    n_jobs = -1,                    
                    g = None, # in case we preload tiles file, for debugging
                    ):
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
            username = input('ASF username: ')
            password = getpass.getpass('ASF password: ')        

    if g is None:
        print ("reading tiles file", flush=True)
        g = gpd.read_file(tiles_file)

    print (f"downloading {len(g)} tiles", flush=True)
    
    mParallel(n_jobs=n_jobs, verbose=30)(
                    delayed(download_job)( 
                        chip                     = chip, 
                        tiles_folder             = tiles_folder, 
                        granules_download_folder = granules_download_folder, 
                        username                 = username, 
                        password                 = password,
                        no_retry                 = no_retry)
                     for _,chip in g.sample(len(g)).iterrows()
            ) 

def download_job( chip, 
                  tiles_folder, 
                  granules_download_folder, 
                  username, 
                  password,
                  no_retry = False):
    
    retry_skipped = not no_retry
    tile = chip.geometry
    dest_file = f"{tiles_folder}/{chip.identifier}.nc"
    skipped_file = f"{tiles_folder}/{chip.identifier}.skipped"

    # if already processed, skip
    if os.path.isfile(dest_file):
        return

    # if attempted previously but failed, retry if requested in case
    # of transitory errors (could not connect, etc.)
    if retry_skipped and os.path.isfile(skipped_file):
        with open(skipped_file) as f:
            errorcode = f.read()
            if errorcode.strip().split(" ")[0] in ['COULD_NOT_DOWNLOAD']:
                pass   # method will continue and tile fownload will be retried
            else:
                return # dont do anything, no retry


    cb = GSSIC_Chip_Builder(chip.identifier, 
                            chip.geometry, 
                            cache_folder=granules_download_folder)
    try:
        cb.download(username, password)
    except Exception as e:
        touch(skipped_file, 'COULD_NOT_DOWNLOAD '+str(e))
        return
    
    try:
        c = cb.get_chip()
    except Exception as e:
        c.close()
        touch(skipped_file, 'COULD_NOT_GET_CHIP '+str(e))
        return
    
    try:
        c.to_netcdf(dest_file)
    except Exception as e:
        c.close()
        touch(skipped_file, 'COULD_NOT_WRITE_CHIP '+str(e))
        return
    
    c.close()
