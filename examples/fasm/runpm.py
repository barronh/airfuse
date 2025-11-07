# %
# Library Imports
# ---------------
import os
import numpy as np
import pandas as pd
import xarray as xr
from airfuse import dnr
from airfuse.layers import naqfc
from airfuse.points import airnowfasm, purpleairfasm
from airfuse.utils import fuse, df2ds, to_geojson

# %
# User Configuration
# ------------------
# - spc : pm25 or ozone
# - nowcast : True or False
# - date : datetime with hour precision
# - yhatsfx : model prediction suffix
#   - mbc_yhatsfx : multiplicative corrected model
#   - abc_yhatsfx : additive corrected model
#   - bc_yhatsfx : average of corrected models after removing negatives
# - cvsfx : suffix for cross-validatin of model
# - ncpath : Path for output to be saved
# - n_jobs : Number of threads to simultaneiously do calculations
spc = 'pm25'
nowcast = True
lag = pd.to_timedelta('1h')
date = (pd.to_datetime('now', utc=True) - lag).floor('1h').tz_convert(None)
yhatsfx = '_gdnr'
cvsfx = f'{yhatsfx}_cv'
ncpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.nc'
n_jobs = 32
fasmcfgpath = '~/fasm.json'

# %
# Perform AirFuse
# ---------------

# Open Model Instance
mod = naqfc(spc, nowcast=nowcast)

# Extract a time-slice layer
modvar = mod.get(date)

# Get observations that match the model space/time coordinates
andf = airnowfasm(spc, nowcast=nowcast, fasmcfgpath=fasmcfgpath).pair(date, modvar, mod.proj)
andf[['groups', 'sample_weight']] = [0, 1]
andf['sample_weight'] = andf['sample_weight'].where(andf['obs'] < 1000, .1)
padf = purpleairfasm(spc, nowcast=nowcast, fasmcfgpath=fasmcfgpath).pair(date, modvar, mod.proj)
padf[['groups', 'sample_weight']] = [1, 0.25]
padf['sample_weight'] = padf['sample_weight'].where(padf['obs'] < 1000, .0025)
padf.query('obs < 1000', inplace=True)
obdf = pd.concat([andf, padf], ignore_index=True)

# Perform Fusion Using Grouped DNR
# - Calculate one surface from pooled weights
# - Weights calculated separately for groups
#   - two base functions,
#   - two Delaunay diagrams and functions
# - sample_weight will be added "automatically to the fitkwds
# - groups will be added "automatically to the fitkwds
gdnr = dnr.GroupedDelaunayNeighborsRegressor(
    delaunay_weights="only", n_neighbors=30,
    weights={
        0: lambda d: np.maximum(1250, d)**-2,
        1: lambda d: np.maximum(2500, d)**-2,
    }, n_jobs=n_jobs
)

# Make Predictions at model centers
tgtdf = modvar.to_dataframe(name='mod')
fuse(tgtdf, obdf, obdnr=gdnr, yhatsfx=yhatsfx, cvsfx=cvsfx)

# %
# Save Outputs
# ------------

# Save the results as a NetCDF file
tgtds = df2ds(
    tgtdf, obdf, yhatsfx=yhatsfx, cvsfx=cvsfx, crs_proj4=modvar.crs_proj4
)
os.makedirs(os.path.dirname(ncpath), exist_ok=True)
tgtds.to_netcdf(ncpath)

# Save the results as a GeoJSON file
colors = [
    '#009500', '#98cb00', '#fefe98', '#fefe00',
    '#fecb00', '#f69800', '#fe0000', '#d50092'
]  # 8 colors between 9 edges
edges = [-5, 10, 20, 30, 50, 70, 90, 120, 1000]
if nowcast:
    colors = ['#00e300', '#fefe00', '#fe7e00', '#fe0000', '#8e3f96', '#7e0023']
    edges = [0, 12, 35.5, 55.5, 150.5, 250.5, 255]  # old pm25 aqi cutpoints EPA 454/B-18-007 September 2018
    edges = [0, 9, 35.5, 55.5, 125.5, 225.5, 255]  # new pm25 aqi cutpoints EPA-454/B-24-002 May 2024

ds = xr.open_dataset(ncpath)
jpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.geojson'
to_geojson(
    jpath, x=ds.x, y=ds.y, z=ds[f'bc{yhatsfx}'][0], crs=ds.crs_proj4,
    edges=edges, colors=colors, under='#eeeeee', over=colors[-1],
    description=ds.description
)
