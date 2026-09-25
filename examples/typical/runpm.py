# %
# Library Imports
# ---------------
import sys
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
nowcast = True
lag = pd.to_timedelta('1.25h')
date = (pd.to_datetime('now', utc=True) - lag).floor('1h').tz_convert(None)
# date = pd.to_datetime('2025-07-15T18')  # Random
# date = pd.to_datetime('2025-01-09T12')  # LA Fires
# date = pd.to_datetime('2025-05-13T08')  # Utah Dust storm (run with ignore, correct, exclude)
dust = 'ignore'
if dust == 'ignore':
    sfx = f'_PM25'
else:
    sfx = f'_PM25_{dust}'
ncpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z{sfx}.nc'
jpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z{sfx}.geojson'
logpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z{sfx}.log'
n_jobs = 32

os.makedirs(os.path.dirname(logpath), exist_ok=True)
# tee out
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=logpath, level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)
logger.info(f'Starting AirFuse {pd.to_datetime("now")}')
logger.info(f'spc={spc}')
logger.info(f'date={date}')
logger.info(f'nowcast={nowcast}')
logger.info(f'ncpath={ncpath}')
logger.info(f'logpath={logpath}')
logger.info(f'n_jobs={n_jobs}')

# %
# Open Model Instance
# -------------------

# Open Model Instance
logger.info('Loading NAQFC')
mod = naqfc(spc, nowcast=nowcast)
modvar = mod.get(date)  # Extract a time-slice layer

# %
# Get observations
# ----------------
# - match the model space/time coordinates
# - AirNow is group 0 with a prior weight of 1
# - PurpleAir is group 1 with a prior weight of 0.25

logger.info('Loading Observations')

logger.info('Loading AirNow')
andf = airnowapi(spc, nowcast=nowcast).pair(date, modvar, mod.proj)
andf[['groups', 'sample_weight']] = [0, 1]
andf['sample_weight'] = andf['sample_weight'].where(andf['obs'] < 1000, .1)
logger.info(f'- AirNow : groups=0 sample_weight=1 n={andf.shape[0]}')

logger.info('Loading PurpleAir')
try:
    # Dynamic list can read from a file
    # import json
    # with open('exclusion.json') as ef:
    #     badids = [int(row['unit_id']) for row in json.load(ef)]
    #
    # Or define a static list
    badids = [
        25795, 36281, 38473, 38649, 79925, 85763, 85915, 110434, 111342,
        111704, 112298, 112484, 113346, 113648, 114217, 114319, 114577,
        117239, 118045, 118579, 118791, 118805, 119545, 119967, 120419,
        120787, 123649, 124621, 127477, 128833, 130447, 131903, 131967,
        132695, 133686, 134212, 137404, 139452, 139978, 142672, 146858,
        148643, 151122, 151748, 152934, 154393, 155609, 155621, 155649,
        160983, 162317, 163203, 164683, 164903, 165113, 165519, 165539,
        165549, 165571, 166809, 166827, 171249, 171711, 175717, 177421,
        180557, 184431, 185427, 186123, 188435, 189167, 190521, 192799,
        192987, 194765, 195355, 195362, 195747, 202913, 203103, 203197,
        227437, 230663, 231387, 231389, 231391, 231393, 231397, 237087,
        237093, 237145, 242071, 242105, 246447, 264472, 270054, 270248,
        270292, 294067, 384473
    ]
    paobj = purpleairrsig(spc, nowcast=nowcast, dust=dust, exclude=badids)
    padf = paobj.pair(date, modvar, mod.proj)
    padf[['groups', 'sample_weight']] = [1, 0.25]
    # downweight samples with obs greater than 1000?
    padf['sample_weight'] = padf['sample_weight'].where(padf['obs'] < 1000, .0025)
    # Or just remove them?
    padf.query('obs < 1000', inplace=True)
    npa = padf.shape[0]
    if npa < 300:
        msg = f'n={npa} insufficient observations for cross-validation'
        raise ValueError(msg)
    logger.info(f'- PurpleAir : groups=1 sample_weight=0.25 n={npa}')
    logger.info('Concatenate AirNow and PurpleAir')
    obdf = pd.concat([andf, padf], ignore_index=True)
except Exception as e:
    msg = f'AirNow only; getting PurpleAir failed: {str(e)}'
    logger.warn(msg)
    obdf = andf
logger.info(f'- Obs : n={obdf.shape[0]}')

# %
# Configure Regressor
# -------------------
# Create a regressor specifying k nieghbors, distance function, and
# parallel processing. Because Ozone has only airnow obs, the weights
# function is simple. All obs within a grid are equally close, so not
# allowing distance closer than 1250m (1/4 of a grid cell) for AirNow
# and no closer than 2500m (1/2 of a grid cell) for PurpleAir.
#
# Perform Fusion Using Grouped DNR
# - Calculate one surface from pooled weights
# - Weights calculated separately for groups
#   - two base functions,
#   - two Delaunay diagrams
# - sample_weight will be added "automatically to the fitkwds
# - groups will be added "automatically" to the fitkwds
anmindist = 1250
pamindist = 2500
logger.info('Configure Grouped DNR')
logger.info(f' - n_neighbors: 30')
logger.info(f' - delaunay_weights: "only"')
logger.info(f' - AirNow Min Dist: {anmindist}m')
logger.info(f' - PurpleAir Min Dist: {pamindist}m')
logger.info(f' - n_jobs: {n_jobs}')
regr = dnr.BCGroupedDelaunayNeighborsRegressor(
    delaunay_weights="only", n_neighbors=30,
    weights={
        0: lambda d: np.maximum(anmindist, d)**-2,
        1: lambda d: np.maximum(pamindist, d)**-2,
    }, n_jobs=n_jobs
)

# %
# Perform Cross validation
# ------------------------
# random_state set for reproducibility
# n_splits using standard 10-fold cross valdiation
# shuffle to ensure order of retrieved records does not affect result
logger.info('Start cross-validation')
kf = KFold(random_state=42, n_splits=10, shuffle=True)
xkeys = ['x', 'y', 'mod']
fitkwds = dict(groups=obdf['groups'], sample_weight=obdf['sample_weight'])
obdf['mod_bbc_cv'] = cross_val_predict(regr, obdf[xkeys], obdf['obs'], cv=kf, params=fitkwds)

# %
# Perform Application
# -------------------
# 1. Fit the full model,
# 2. Predict at observational locations
# 3. Predict at target locations

regr.fit(obdf[xkeys], obdf['obs'], **fitkwds)
obdf['mod_bbc'] = regr.predict(obdf[xkeys])

tgtdf = modvar.to_dataframe(name='mod')
tgtX = tgtdf.index.to_frame()[['x', 'y']]
tgtX['mod'] = tgtdf['mod']
regr.set_how('debug')  # save space by using all or best
tgtdf[regr.feature_names_out_] = regr.predict(tgtX)

# %
# Save Outputs
# ------------

# Save the results as a NetCDF file
logger.info('Saving result as NetCDF')

# Convert outputs from 64-bit to 32-bit floats to save space.
outtypes = {k: np.float32 for k in regr.feature_names_out_}
outtypes['mod'] = np.float32
tgtds = tgtdf.astype(outtypes).to_xarray()
tgtds['obsx'] = obdf['x'].to_xarray()
tgtds['obsy'] = obdf['y'].to_xarray()
tgtds['obs'] = obdf['obs'].to_xarray()
tgtds['groups'] = obdf['groups'].to_xarray()
tgtds['sample_weight'] = obdf['sample_weight'].to_xarray()
tgtds['mod_bbc_cv'] = obdf['mod_bbc_cv'].to_xarray()
tgtds['mod'].attrs.update(modvar.attrs)
addattrs(tgtds, units=modvar.units)
tgtds.attrs['crs_proj4'] = modvar.crs_proj4
tgtds.rename(index='obsn').to_netcdf(ncpath)

logger.info(f'Completed AirFuse {pd.to_datetime("now")}')

# Save the results as a GeoJSON file
logger.info('Saving result as GeoJson')
inf = float('inf')
if nowcast:
    # EPA AQI Color Scale
    colors = [
        '#eeeeee', '#00e300', '#fefe00', '#fe7e00', '#fe0000', '#8e3f96',
        '#7e0023', '#7e0023'
    ]
    # old pm25 aqi cutpoints EPA 454/B-18-007 September 2018
    edges = [-inf, 0, 12, 35.5, 55.5, 150.5, 250.5, 255, inf]
    # new pm25 aqi cutpoints EPA-454/B-24-002 May 2024
    edges = [-inf, 0, 9, 35.5, 55.5, 125.5, 225.5, 255, inf]
else:
    # AirNowTech 1h Color Scale as of 2026-06-25
    colors = [
        '#c8ffc8', '#00e400', '#007d00', '#ffffc8', '#ffff00', '#c8c800',
        '#ffbe78', '#ff7e00', '#c86400', '#ff6464', '#ff0000', '#c80000',
        '#c896c8', '#8f3f97', '#643264', '#7e0023', '#500019', '#32000f',
        '#000000'
    ]
    edges = [
        -inf, 3.0, 6.0, 9.1, 15.0, 25.0, 35.5, 40.0, 50.0, 55.5, 75.0, 100.0,
        125.5, 150.0, 200.0, 225.5, 325.0, 500.0, 750.0, inf
    ]

to_geojson(
    jpath, x=tgtds.x, y=tgtds.y, z=tgtds['mod_bbc'][0], crs=tgtds.crs_proj4,
    edges=edges, colors=colors, description=tgtds.description
)
