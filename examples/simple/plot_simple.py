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
from airfuse.utils import mpestats
from airfuse import dnr


# %
# User Configuration
# ------------------
# - spc : pm25 or ozone
# - nowcast : True or False
# - date : datetime with hour precision
# - ncpath : Path for output to be saved
spc = 'pm25'
nowcast = False
date = pd.to_datetime('2025-01-09T12')
figpath = f'outputs/{date:%Y%m%d/AirFuse.%Y-%m-%dT%H}Z_PM25'


os.makedirs(os.path.dirname(figpath), exist_ok=True)

# %
# Perform AirFuse
# ---------------

# Open Model Instance
mod = naqfc(spc, nowcast=nowcast)

# Extract a time-slice layer
modvar = mod.get(date)

# Get observations that match the model space/time coordinates
obdf = airnowapi(spc, nowcast=nowcast).pair(date, modvar, mod.proj)

# Create a regressor specifying k nieghbors, distance function, and
# parallel processing.
regr = dnr.BCDelaunayNeighborsRegressor(
    n_neighbors=30, weights=lambda d: np.maximum(d, 1e-10)**-2
)

# Perform Cross validation
kf = KFold(random_state=42, n_splits=10, shuffle=True)
xkeys = ['x', 'y', 'mod']
obdf['mod_bc_cv'] = cross_val_predict(regr, obdf[xkeys], obdf['obs'], cv=kf)

# Fit the full model
regr.fit(obdf[xkeys], obdf['obs'])
obdf['mod_bc'] = regr.predict(obdf[xkeys])

# %
# Make Performance Plots
# ----------------------

units = 'micrograms/m**3'
yhatkeys = ['mod', 'mod_bc_cv', 'mod_bc']
cvdf = obdf[['obs'] + yhatkeys]
statdf = mpestats(cvdf)
for yhatkey in yhatkeys:
    vmin =  statdf.loc['min', ['obs', yhatkey]].min()
    vmax =  statdf.loc['max', ['obs', yhatkey]].max()
    skeys = ['count', 'mean', 'mb', 'me', 'rmse', 'r']
    label = '\n'.join(str(statdf[yhatkey][skeys].round(2)).split('\n')[:-1])
    ax = cvdf.plot.hexbin(
        x='obs', y=yhatkey, gridsize=100, extent=(vmin, vmax, vmin, vmax),
        mincnt=1, cmap='viridis'
    )
    ax.set(xlabel=f'obs [{units}]', ylabel=f'{yhatkey} [{units}]')
    ax.text(vmax, vmin, label, ha='right', bbox=dict(facecolor='w', edgecolor='k'))
    ax.figure.savefig(figpath + '_' + yhatkey + '.png')
