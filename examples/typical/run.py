"""
Apply AirFuse Over US
=====================

This script can be called to fuse AirNow and PurpleAir with NOAA's NAQFC
model to create a best estimate of pm25 over the US. Then, create GeoJson
and png files that can be used with leaflet to create an operational server.
The map.html file is set to point to EPA's AWS s3 bucket of live results,
but can be changed to point to your own public data (on AWS or any https).

Save the code as run.py and call the script with the bash commands:
.. code::bash

    # source airfuse/venv/bin/activate
    python run.py -j 4 o3
    python run.py -j 4 pm25

"""
from airfuse.drivers import fuse
from airfuse.pm import pmfuse
from airfuse.util import to_geojson, to_webpng
from airfuse import style
import pandas as pd
import xarray as xr
import argparse

defdate = (pd.to_datetime('now', utc=True) - pd.to_timedelta('1.416h'))
defdate = defdate.floor('1h')
prsr = argparse.ArgumentParser()
prsr.add_argument('-O', '--overwrite', default=False, action='store_true')
prsr.add_argument('-j', '--njobs', default=None, type=int)
prsr.add_argument('spc')
prsr.add_argument('sdate', nargs='?', default=defdate, type=pd.to_datetime)
args = prsr.parse_args()

# Common options
opts = dict(
    startdate=args.sdate, model='naqfc', bbox=(-135, 15, -55, 60),
    cv_only=False, overwrite=args.overwrite, format='nc', njobs=args.njobs
)

# Setup species specific options
if args.spc == 'pm25':
    fusefunc = pmfuse
    fusedkey = 'FUSED_aVNA'
    colors = style.ant1hpmcolors
    edges = style.ant1hpmedges
    norm = style.ant_1hpm_norm
    cmap = style.ant_1hpm_cmap
elif args.spc == 'o3':
    fusefunc = fuse
    # regular driver requires obssource, species, startdate, model...
    opts['obssource'] = 'airnow'
    opts['species'] = args.spc
    fusedkey = 'aVNA'
    colors = style.ant1ho3colors
    edges = style.ant1ho3edges
    norm = style.ant_1ho3_norm
    cmap = style.ant_1ho3_cmap

# Perform datafusion
outpaths = fusefunc(**opts)

# Retrieve NetCDF results
ds = xr.open_dataset(outpaths['outpath'])

# Write out a copy as a contour geojson
jsonpath = outpaths['outpath'].replace('.nc', '.geojson')
to_geojson(
    jsonpath, ds.x, ds.y, ds[fusedkey][0], ds.crs_proj4,
    edges=edges, colors=colors, under=colors[0], over=colors[-1],
    description=ds.description
)

# Write out a copy as a WebMercator Raster PNG
pngpath = outpaths['outpath'].replace('.nc', '.png')
to_webpng(
    ds, bbox=None, dx=2000., dy=2000., key=fusedkey, pngpath=pngpath,
    norm=norm, cmap=cmap
)
