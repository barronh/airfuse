"""
Apply AirFuse Over Southwest US
===============================

Use AirNow monitors and NOAA's NAQFC model to create a best estimate of
ozone over the southwest US. The results include cross-validation and
a gridded NetCDF output. Both are plotted.
"""
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from airfuse.drivers import fuse
from airfuse import style
from airfuse.util import mpestats
import pycno

# %%
# Perform the fusion
# ------------------
# Fuse AirNow ozone monitors w/ NOAA Air Quality Forecast Capability (naqfc)
# Set the datetime with hour specificity (17-17:59Z) and use a window
# (alternatives: replace airnow with aqs and/or naqfc with geoscf)
#
outpaths = fuse(
    obssource='airnow', species='o3', model='naqfc',
    startdate='2025-04-14T17Z', bbox=(-120, 25, -97, 35),  # Focus time/area
    format='nc'
)

# %%
# Display Results
# ---------------
#
ds = xr.open_dataset(outpaths['outpath'])
cvdf = pd.read_csv(outpaths['evalpath'])

# Print Cross-Validation Statistics
statdf = mpestats(cvdf[['ozone', 'CV_aVNA']], refkey='ozone')
print(statdf.T.round(2).to_string())

# %%
# Plot Cross-Validation
fig, ax = plt.subplots()
opts = dict(mincnt=1, gridsize=40)
cvdf.plot.hexbin(x='ozone', y='CV_aVNA', ax=ax, **opts)
lim = (0, cvdf[['ozone', 'CV_aVNA']].max().max() * 1.1)
ax.set(facecolor='gainsboro', xlim=lim, ylim=lim, xlabel='AirNow [ppb]', ylabel='AirFuse CV [ppb]')
label = statdf.loc['CV_aVNA', ['nmb%', 'r', 'rmse%']].T.round(2).to_string()
ax.text(0.05, 0.975, label, va='top', ha='left', font='monospace', transform=ax.transAxes)
fig.savefig(outpaths['evalpath'] + '.png')

# %%
# Plot Map
opts = dict(cmap=style.ant_1ho3_cmap, norm=style.ant_1ho3_norm)
fig, axx = plt.subplots(1, 2, figsize=(12, 4))
qm = axx[0].pcolormesh(ds.x, ds.y, ds['aVNA'][0], **opts)
fig.colorbar(qm, label='AirFuse [ppb]', extend='both')
qm = axx[1].pcolormesh(ds.x, ds.y, ds['IDW_ozone'][0], **opts)
fig.colorbar(qm, label='Inverse Distance Weighted [ppb]', extend='both')
pycno.cno(ds.crs_proj4).drawstates(ax=axx)
fig.savefig(outpaths['outpath'] + '.png')
