# Tiling ARIA Sentinel-1 Geocoded Unwrapped Interferograms

https://asf.alaska.edu/data-sets/derived-data-sets/sentinel-1-interferograms/

```
-----------------------------------------------------------
JPL's Aria GUNW dataset extractor utility 0.1.dev1+g9f91d9e.d20230528
-----------------------------------------------------------

usage: gunwt tiles2granules [-h] --tiles_file TILES_FILE --tiles_folder TILES_FOLDER
                            --granules_download_folder GRANULES_DOWNLOAD_FOLDER --year YEAR --month
                            MONTH [--no_retry] [--n_jobs N_JOBS]

options:
  -h, --help            show this help message and exit
  --tiles_file TILES_FILE
                        geopandas dataframe containing tiles, with 'geometry' and 'identifier' columns
  --tiles_folder TILES_FOLDER
                        where to store the resulting tiles
  --granules_download_folder GRANULES_DOWNLOAD_FOLDER
                        where to download the granules from ASF before tiling them
  --year YEAR           year to query
  --month MONTH         month to query
  --no_retry            if set, skipped tiles in previous runs will not be retried
  --n_jobs N_JOBS       number of parallel jobs (defaults to -1, using all CPUs)

```

