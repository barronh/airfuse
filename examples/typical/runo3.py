# %
# Library Imports
# ---------------
import os
import pandas as pd
import xarray as xr
from airfuse.layers import naqfc
from airfuse.points import airnowapi
from airfuse.utils import fuse, df2ds, to_geojson
import logging

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
spc = 'ozone'
nowcast = False
# date = pd.to_datetime('2025-01-09T12')
lag = pd.to_timedelta('1h')
date = (pd.to_datetime('now', utc=True) - lag).floor('1h').tz_convert(None)
yhatsfx = '_andnr'
cvsfx = f'{yhatsfx}_cv'
ncpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.nc'
logpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.log'
n_jobs = 32

os.makedirs(os.path.dirname(logpath), exist_ok=True)
logging.basicConfig(filename=logpath, level=logging.INFO)
logging.info('Starting AirFuse')
logging.info(f'spc={spc}')
logging.info(f'date={date}')
logging.info(f'nowcast={nowcast}')
logging.info(f'ncpath={ncpath}')
logging.info(f'logpath={logpath}')
logging.info(f'n_jobs={n_jobs}')

# %
# Perform AirFuse
# ---------------

# Open Model Instance
logging.info('Loading NAQFC')
mod = naqfc(spc, nowcast=nowcast)

# Extract a time-slice layer
modvar = mod.get(date)

# Get observations that match the model space/time coordinates
logging.info('Loading AirNow')
obdf = airnowapi(spc, nowcast=nowcast).pair(date, modvar, mod.proj)

# Make Predictions at model centers
logging.info('Start fitting, cross-validation, and predictions')
tgtdf = modvar.to_dataframe(name='mod')
fuse(tgtdf, obdf, yhatsfx=yhatsfx, cvsfx=cvsfx, dnrkwds=dict(n_jobs=n_jobs))

# %
# Save Outputs
# ------------

# Save the results as a NetCDF file
logging.info('Saving result as NetCDF')
tgtds = df2ds(
    tgtdf, obdf, yhatsfx=yhatsfx, cvsfx=cvsfx, crs_proj4=modvar.crs_proj4
)
tgtds.to_netcdf(ncpath)

# Save the results as a GeoJSON file
logging.info('Saving result as GeoJSON')
colors = [
    '#00fe00', '#fefe80', '#fefe00', '#fbbe43', '#fe8000', '#fe0000'
]  # 6 colors beteen 7 edges
edges = [0, 60, 80, 100, 112, 125, 1000]
if nowcast:
    colors = ['#00e300', '#fefe00', '#fe7e00', '#fe0000', '#8e3f96', '#7e0023']
    edges = [0, 54, 70, 85, 105, 200, 255]  # ozone aqi cutpoints

ds = xr.open_dataset(ncpath)
jpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.geojson'
to_geojson(
    jpath, x=ds.x, y=ds.y, z=ds[f'bc{yhatsfx}'][0], crs=ds.crs_proj4,
    edges=edges, colors=colors, under='#eeeeee', over=colors[-1],
    description=ds.description
)
