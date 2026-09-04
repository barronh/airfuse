# %
# Library Imports
# ---------------
import os
import numpy as np
import pandas as pd
import logging

from airfuse.layers import naqfc
from airfuse.points import airnowapi
from airfuse.utils import addattrs, to_geojson
from airfuse import dnr
# Used for Cross-Validation
from sklearn.model_selection import KFold, cross_val_predict

# %
# User Configuration
# ------------------
# - spc : pm25 or ozone
# - nowcast : True or False
# - date : datetime with hour precision
# - ncpath : Path for output to be saved
# - n_jobs : Number of threads to simultaneiously do calculations
spc = 'ozone'
nowcast = True
# date = pd.to_datetime('2025-01-09T12')
lag = pd.to_timedelta('1h')
date = (pd.to_datetime('now', utc=True) - lag).floor('1h').tz_convert(None)
ncpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z_Ozone.nc'
jpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z_Ozone.geojson'
logpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z_Ozone.log'
n_jobs = 32

os.makedirs(os.path.dirname(logpath), exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(filename=logpath, level=logging.INFO)
logging.info('Starting AirFuse')
logging.info(f'spc={spc}')
logging.info(f'date={date}')
logging.info(f'nowcast={nowcast}')
logging.info(f'ncpath={ncpath}')
logging.info(f'logpath={logpath}')
logging.info(f'n_jobs={n_jobs}')

# %
# Open Model Instance
# -------------------

logging.info('Loading NAQFC')
mod = naqfc(spc, nowcast=nowcast)
modvar = mod.get(date)  # # Extract a time-slice layer

# %
# Get observations
# ----------------
# match the model space/time coordinates

logging.info('Loading AirNow')
obdf = airnowapi(spc, nowcast=nowcast).pair(date, modvar, mod.proj)
logger.info(f'- AirNow : groups=0 sample_weight=1 n={obdf.shape[0]}')

# %
# Configure Regressor
# -------------------
# Create a regressor specifying k nieghbors, distance function, and
# parallel processing. Because Ozone has only airnow obs, the weights
# function is simple. All obs within a grid are equally close, so not
# allowing distance closer than 1250m (1/4 of a grid cell)

anmindist = 1250 # approximately 1/4 grid cell in meters 
regr = dnr.BCDelaunayNeighborsRegressor(
    n_jobs=n_jobs, n_neighbors=30,
    weights=lambda d: np.maximum(d, anmindist)**-2, how='debug'
)

# %
# Perform Cross validation
# ------------------------
# random_state set for reproducibility
# n_splits using standard 10-fold cross valdiation
# shuffle to ensure order of retrieved records does not affect result
logging.info('Start cross-validation')
kf = KFold(random_state=42, n_splits=10, shuffle=True)
xkeys = ['x', 'y', 'mod']
obdf['mod_bbc_cv'] = cross_val_predict(regr, obdf[xkeys], obdf['obs'], cv=kf)[:, -1]

# %
# Perform Application
# -------------------
# 1. Fit the full model,
# 2. Predict at observational locations
# 3. Predict at target locations

logging.info('Start obs application')
regr.fit(obdf[xkeys], obdf['obs'])
obdf['mod_bbc'] = regr.predict(obdf[xkeys])[:, -1]
logging.info('Start target application')
tgtdf = modvar.to_dataframe(name='mod')
tgtX = tgtdf.index.to_frame()[['x', 'y']]
tgtX['mod'] = tgtdf['mod']
pred = regr.predict_dataframe(tgtX)
for key in pred.columns:
    tgtdf[key] = pred[key].values

# %
# Save Outputs
# ------------

# Save the results as a NetCDF file
logging.info('Saving result as NetCDF')

tgtds = tgtdf.to_xarray()
tgtds['obsx'] = obdf['x'].to_xarray()
tgtds['obsy'] = obdf['y'].to_xarray()
tgtds['obs'] = obdf['obs'].to_xarray()
tgtds['mod_bbc_cv'] = obdf['mod_bbc_cv'].to_xarray()
tgtds['mod'].attrs.update(modvar.attrs)
addattrs(tgtds, units=modvar.units)
tgtds.attrs['crs_proj4'] = modvar.crs_proj4
tgtds.to_netcdf(ncpath)

# Save the results as a GeoJSON file
logging.info('Saving result as GeoJSON')

inf = float('inf')
if nowcast:
    # EPA AQI Color Scale
    colors = [
        '#eeeeee', '#00e300', '#fefe00', '#fe7e00', '#fe0000', '#8e3f96',
        '#7e0023', '#7e0023'
    ]
    edges = [-inf, 0, 54, 70, 85, 105, 200, 255, inf]  # ozone aqi cutpoints
else:
    # AirNowTech 1h Color Scale as of 2026-06-25
    colors = [
        '#c8ffc8', '#00e400', '#007d00', '#ffffc8', '#ffff00', '#c8c800',
        '#ffbe78', '#ff7e00', '#c86400', '#ff6464', '#ff0000', '#c80000',
        '#c896c8', '#8f3f97', '#643264', '#000000'
    ]
    edges = [
        -inf, 30, 45, 55, 60, 65, 71, 75, 80, 86, 90, 100, 106, 125, 175, 201,
        inf
    ]

to_geojson(
    jpath, x=tgtds.x, y=tgtds.y, z=tgtds['mod_bbc'][0], crs=tgtds.crs_proj4,
    edges=edges, colors=colors,
    description=tgtds.description
)
