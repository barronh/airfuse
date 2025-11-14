# %
# Library Imports
# ---------------
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from airfuse import dnr
from airfuse.layers import naqfc
from airfuse.points import airnowapi, purpleairrsig
from airfuse.utils import addattrs, to_geojson
import logging

# %
# User Configuration
# ------------------
# - spc : pm25 or ozone
# - nowcast : True or False
# - date : datetime with hour precision
# - ncpath : Path for output to be saved
# - n_jobs : Number of threads to simultaneiously do calculations
spc = 'pm25'
nowcast = False
lag = pd.to_timedelta('1h')
date = (pd.to_datetime('now', utc=True) - lag).floor('1h').tz_convert(None)
# date = pd.to_datetime('2025-01-09T12')
ncpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z_PM25.nc'
jpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z_PM25.geojson'
logpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z_PM25.log'
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

logging.info('Loading Observations')
logging.info('- AirNow : groups=0 sample_weight=1')
# Get observations that match the model space/time coordinates
andf = airnowapi(spc, nowcast=nowcast).pair(date, modvar, mod.proj)
andf[['groups', 'sample_weight']] = [0, 1]
andf['sample_weight'] = andf['sample_weight'].where(andf['obs'] < 1000, .1)

logging.info('- PurpleAir : groups=1 sample_weight=0.25')
padf = purpleairrsig(spc, nowcast=nowcast).pair(date, modvar, mod.proj)
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
anmindist = 1250
pamindist = 2500
logging.info('Configure Grouped DNR')
logging.info(f' - AirNow Min Dist: {anmindist}m')
logging.info(f' - PurpleAir Min Dist: {pamindist}m')
regr = dnr.BCGroupedDelaunayNeighborsRegressor(
    delaunay_weights="only", n_neighbors=30,
    weights={
        0: lambda d: np.maximum(anmindist, d)**-2,
        1: lambda d: np.maximum(pamindist, d)**-2,
    }, n_jobs=n_jobs
)

# Make Predictions at model centers
logging.info('Start fitting, cross-validation, and predictions')
kf = KFold(random_state=42, n_splits=10, shuffle=True)
xkeys = ['x', 'y', 'mod']
fitkwds = dict(groups=obdf['groups'], sample_weight=obdf['sample_weight'])
obdf['mod_bc_cv'] = cross_val_predict(regr, obdf[xkeys], obdf['obs'], cv=kf, params=fitkwds)

# Fit the full model
regr.fit(obdf[xkeys], obdf['obs'], **fitkwds)
obdf['mod_bc'] = regr.predict(obdf[xkeys])

tgtdf = modvar.to_dataframe(name='mod')
tgtX = tgtdf.index.to_frame()[['x', 'y']]
tgtX['mod'] = tgtdf['mod']
tgtdf['mod_bc'] = regr.predict(tgtX)

# %
# Save Outputs
# ------------

# Save the results as a NetCDF file
logging.info('Saving result as NetCDF')

tgtds = tgtdf.to_xarray()
tgtds['obsx'] = obdf['x'].to_xarray()
tgtds['obsy'] = obdf['y'].to_xarray()
tgtds['obs'] = obdf['obs'].to_xarray()
tgtds['groups'] = obdf['groups'].to_xarray()
tgtds['sample_weight'] = obdf['sample_weight'].to_xarray()
tgtds['mod_bc_cv'] = obdf['mod_bc_cv'].to_xarray()
tgtds['mod'].attrs.update(modvar.attrs)
addattrs(tgtds, units=modvar.units)
tgtds.attrs['crs_proj4'] = modvar.crs_proj4
tgtds.to_netcdf(ncpath)

# Save the results as a GeoJSON file
logging.info('Saving result as GeoJson')
colors = [
    '#009500', '#98cb00', '#fefe98', '#fefe00',
    '#fecb00', '#f69800', '#fe0000', '#d50092'
]  # 8 colors between 9 edges
edges = [-5, 10, 20, 30, 50, 70, 90, 120, 1000]

if nowcast:
    colors = ['#00e300', '#fefe00', '#fe7e00', '#fe0000', '#8e3f96', '#7e0023']
    # old pm25 aqi cutpoints EPA 454/B-18-007 September 2018
    edges = [0, 12, 35.5, 55.5, 150.5, 250.5, 255]
    # new pm25 aqi cutpoints EPA-454/B-24-002 May 2024
    edges = [0, 9, 35.5, 55.5, 125.5, 225.5, 255]

to_geojson(
    jpath, x=tgtds.x, y=tgtds.y, z=tgtds['mod_bc'][0], crs=tgtds.crs_proj4,
    edges=edges, colors=colors, under='#eeeeee', over=colors[-1],
    description=tgtds.description
)
