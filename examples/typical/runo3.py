# %
# Library Imports
# ---------------
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from airfuse.layers import naqfc
from airfuse.points import airnowapi
from airfuse.utils import addattrs, to_geojson
from airfuse import dnr
import logging

# %
# User Configuration
# ------------------
# - spc : pm25 or ozone
# - nowcast : True or False
# - date : datetime with hour precision
# - ncpath : Path for output to be saved
# - n_jobs : Number of threads to simultaneiously do calculations
spc = 'ozone'
nowcast = False
# date = pd.to_datetime('2025-01-09T12')
lag = pd.to_timedelta('2h')
date = (pd.to_datetime('now', utc=True) - lag).floor('1h').tz_convert(None)
ncpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z_Ozone.nc'
jpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z_Ozone.geojson'
logpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z_Ozone.log'
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
print(obdf.shape[0])
# Create a regressor specifying k nieghbors, distance function, and
# parallel processing.
regr = dnr.BCDelaunayNeighborsRegressor(
    n_jobs=n_jobs, n_neighbors=30,
    weights=lambda d: np.maximum(d, 1e-10)**-2
)

logging.info('Start fitting and cross-validation')

# Perform Cross validation
kf = KFold(random_state=42, n_splits=10, shuffle=True)
xkeys = ['x', 'y', 'mod']
obdf['mod_bc_cv'] = cross_val_predict(regr, obdf[xkeys], obdf['obs'], cv=kf)

# Fit the full model
regr.fit(obdf[xkeys], obdf['obs'])
obdf['mod_bc'] = regr.predict(obdf[xkeys])

logging.info('Start model applicaiton')
# Make Predictions at model centers
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
tgtds['mod_bc_cv'] = obdf['mod_bc_cv'].to_xarray()
tgtds['mod'].attrs.update(modvar.attrs)
addattrs(tgtds, units=modvar.units)
tgtds.attrs['crs_proj4'] = modvar.crs_proj4
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

to_geojson(
    jpath, x=tgtds.x, y=tgtds.y, z=tgtds['mod_bc'][0], crs=tgtds.crs_proj4,
    edges=edges, colors=colors, under='#eeeeee', over=colors[-1],
    description=tgtds.description
)
