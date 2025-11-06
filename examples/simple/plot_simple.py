# %
# Library Imports
# ---------------
import os
import pandas as pd
import xarray as xr
from airfuse.layers import naqfc
from airfuse.points import airnowapi
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
spc = 'pm25'
nowcast = False
date = pd.to_datetime('2025-01-09T12')
yhatsfx = '_andnr'
cvsfx = f'{yhatsfx}_cv'
ncpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.nc'

# %
# Perform AirFuse
# ---------------

# Open Model Instance
mod = naqfc(spc, nowcast=nowcast)

# Extract a time-slice layer
modvar = mod.get(date)

# Get observations that match the model space/time coordinates
obdf = airnowapi(spc, nowcast=nowcast).pair(date, modvar, mod.proj)

# Make Predictions at model centers
tgtdf = modvar.to_dataframe(name='mod')
fuse(tgtdf, obdf, yhatsfx=yhatsfx, cvsfx=cvsfx)

# %
# Save Outputs
# ------------

# Save the results as a NetCDF file
tgtdf.set_index('time', append=True, inplace=True)
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
ds = xr.open_dataset(ncpath)
jpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.geojson'
to_geojson(
    jpath, x=ds.x, y=ds.y, z=ds[f'bc{yhatsfx}'][0], crs=ds.crs_proj4,
    edges=edges, colors=colors, under='#eeeeee', over=colors[-1],
    description=ds.description
)
