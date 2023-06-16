# SAR datasets download and tiling utility


```
-----------------------------------------------------------
SAR datasets download and tiling utility 0.1.dev38+g8a440bb.d20230607
-----------------------------------------------------------
usage: sart [-h] {gunw,gssic,chop} ...

options:
  -h, --help         show this help message and exit

commands:
  {gunw,gssic,chop}
    gunw             downloads tiles for JPLs GUNW data collection, from ASF
    gssic            downloads tiles for Global Seasonal Coherence collection, from ASF
    chop             chips any xarray readable (netcfd, tif) file into tiles

```

## For JPL Global Unwrapped Interferometry (GUNW)

See https://asf.alaska.edu/data-sets/derived-data-sets/sentinel-1-interferograms/

```
usage: sart gunw [-h] --tiles_file TILES_FILE --tiles_folder TILES_FOLDER --granules_download_folder
                 GRANULES_DOWNLOAD_FOLDER --year YEAR --month MONTH [--no_retry] [--n_jobs N_JOBS]

options:
  -h, --help            show this help message and exit
  --tiles_file TILES_FILE
                        geopandas dataframe containing tiles, with 'geometry' and 'identifier'
                        columns
  --tiles_folder TILES_FOLDER
                        where to store the resulting tiles
  --granules_download_folder GRANULES_DOWNLOAD_FOLDER
                        where to download the granules from ASF before tiling them
  --year YEAR           year to query
  --month MONTH         month to query
  --no_retry            if set, skipped tiles in previous runs will not be retried
  --n_jobs N_JOBS       number of parallel jobs (defaults to -1, using all CPUs)
```

# For Global Seasonal Senintel-1 Coherence (GSSIC)

See https://asf.alaska.edu/datasets/derived/global-seasonal-sentinel-1-interferometric-coherence-and-backscatter-dataset/
```
usage: sart gssic [-h] --tiles_file TILES_FILE --tiles_folder TILES_FOLDER --granules_download_folder
                  GRANULES_DOWNLOAD_FOLDER [--no_retry] [--n_jobs N_JOBS]

options:
  -h, --help            show this help message and exit
  --tiles_file TILES_FILE
                        geopandas dataframe containing tiles, with 'geometry' and 'identifier'
                        columns
  --tiles_folder TILES_FOLDER
                        where to store the resulting tiles
  --granules_download_folder GRANULES_DOWNLOAD_FOLDER
                        where to download the granules from ASF before tiling them
  --no_retry            if set, skipped tiles in previous runs will not be retried
  --n_jobs N_JOBS       number of parallel jobs (defaults to -1, using all CPUs)

```

