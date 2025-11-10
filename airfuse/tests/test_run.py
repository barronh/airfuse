import os
import pytest
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
_haspakey = os.path.exists(os.path.expanduser('~/.purpleairkey'))

_debug = False
# import matplotlib.colors as mc
# colors = [
#     [0, 150, 0], [153, 204, 0], [255, 255, 153], [255, 255, 0],
#     [255, 204, 0], [247, 153, 0], [255, 0, 0], [214, 0, 147],
# ]
# colors = [mc.to_hex(c) for c in np.array(colors) / 256]
_colors = [
    '#009500',
    '#98cb00', '#fefe98', '#fefe00', '#fecb00', '#f69800', '#fe0000',
    '#d50092'
]
_edges = [-5, 10, 20, 30, 50, 70, 90, 120, 1000]
spc = 'pm25'
date = pd.to_datetime('2025-01-09T11')
ylim = (1.15e6, 1.25e6)
xlim = (-2.22e6, -2.1e6)
nowcast = False


@pytest.mark.skipif(_haspakey, reason="running fullexample")
def test_airnowonly():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_predict
    from ..layers import naqfc
    from ..points import airnowrsig
    from ..utils import addattrs, to_geojson
    from .. import dnr

    with tempfile.TemporaryDirectory() as td:
        ncpath = f'{td}/{date:%Y/%m/%d/AirFuse.%Y-%m-%dT%H}Z.nc'
        jpath = f'{td}/{date:%Y/%m/%d/AirFuse.%Y-%m-%dT%H}Z.geojson'
        mod = naqfc(spc, nowcast=nowcast, inroot=td)
        # get acquires a local cache for reuse in nowcast
        modvar = mod.get(date)

        # Get Obs: AirNow & PurpleAir
        xkeys = ['x', 'y', 'mod']
        ykeys = ['obs']
        obdf = airnowrsig(spc, nowcast=nowcast, inroot=td).pair(
            date, modvar, mod.proj
        )[xkeys + ykeys]
        obdf['groups'] = 'an'
        obdf['sample_weight'] = 1.

        # Perform Fusion Using Grouped DNR
        # - Calculate one surface from pooled weights
        # - Weights calculated separately for groups
        #   - two base functions,
        #   - two Delaunay diagrams and functions
        regr = dnr.BCGroupedDelaunayNeighborsRegressor(
            n_neighbors=30, delaunay_weights='only', weights={
                'an': lambda d: np.maximum(1250, d)**-2,
                'pa': lambda d: np.maximum(2500, d)**-2,
            }
        )
        fitkwds = dict(
            sample_weight=obdf['sample_weight'], groups=obdf['groups']
        )
        tgtdf = modvar.sel(
            x=slice(*xlim), y=slice(*ylim)
        ).to_dataframe(name='mod')
        tgtX = tgtdf.index.to_frame()[['x', 'y']]
        tgtX['mod'] = tgtdf['mod']
        kf = KFold(random_state=42, n_splits=10, shuffle=True)
        xkeys = ['x', 'y', 'mod']
        fitkwds = dict(
            groups=obdf['groups'], sample_weight=obdf['sample_weight']
        )
        obdf['mod_bc_cv'] = cross_val_predict(
            regr, obdf[xkeys], obdf['obs'], cv=kf, params=fitkwds
        )
        regr.fit(obdf[xkeys], obdf['obs'], **fitkwds)
        tgtdf['mod_bc'] = regr.predict(tgtX)

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
        os.makedirs(os.path.dirname(ncpath), exist_ok=True)
        tgtds.to_netcdf(ncpath)

        ds = xr.open_dataset(ncpath)
        to_geojson(
            jpath, x=ds.x, y=ds.y, z=ds['mod_bc'][0], crs=ds.crs_proj4,
            edges=_edges, colors=_colors, under=_colors[0], over=_colors[-1],
            description=ds.description
        )
        if _debug:
            os.system(f'cp {jpath} ./')
            os.system(f'cp {ncpath} ./')


@pytest.mark.skipif(not _haspakey, reason="requires ~/.purpleairkey")
def test_fullexample():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_predict
    from ..layers import naqfc
    from ..points import airnowrsig, purpleairrsig
    from ..utils import addattrs, to_geojson
    from .. import dnr
    with tempfile.TemporaryDirectory() as td:
        ncpath = f'{td}/{date:%Y/%m/%d/AirFuse.%Y-%m-%dT%H}Z.nc'
        jpath = f'{td}/{date:%Y/%m/%d/AirFuse.%Y-%m-%dT%H}Z.geojson'

        mod = naqfc(spc, nowcast=nowcast, inroot=td)
        # get acquires a local cache for reuse in nowcast
        modvar = mod.get(date)

        # Get Obs: AirNow & PurpleAir
        andf = airnowrsig(spc, nowcast=nowcast, inroot=td).pair(
            date, modvar, mod.proj
        )
        padf = purpleairrsig(spc, nowcast=nowcast, inroot=td).pair(
            date, modvar, mod.proj
        )
        xkeys = ['x', 'y', 'mod']
        ykeys = ['obs']
        obdf = pd.concat([andf, padf], keys=['an', 'pa'])[xkeys + ykeys]
        obdf.index.names = ['groups', 'id']
        obdf = obdf.reset_index()
        obdf['sample_weight'] = np.where(
            obdf['groups'] == 'an',
            1, (obdf['obs'] < 1000).astype('i') * 0.25  # weight to 0 if > 1000
        )

        # Perform Fusion Using Grouped DNR
        # - Calculate one surface from pooled weights
        # - Weights calculated separately for groups
        #   - two base functions,
        #   - two Delaunay diagrams and functions
        regr = dnr.BCGroupedDelaunayNeighborsRegressor(
            n_neighbors=30, delaunay_weights='only', weights={
                'an': lambda d: np.maximum(1250, d)**-2,
                'pa': lambda d: np.maximum(2500, d)**-2,
            }
        )
        fitkwds = dict(
            sample_weight=obdf['sample_weight'], groups=obdf['groups']
        )
        tgtdf = modvar.sel(
            x=slice(*xlim), y=slice(*ylim)
        ).to_dataframe(name='mod')
        tgtX = tgtdf.index.to_frame()[['x', 'y']]
        tgtX['mod'] = tgtdf['mod']
        kf = KFold(random_state=42, n_splits=10, shuffle=True)
        xkeys = ['x', 'y', 'mod']
        fitkwds = dict(
            groups=obdf['groups'],
            sample_weight=obdf['sample_weight']
        )
        obdf['mod_bc_cv'] = cross_val_predict(
            regr, obdf[xkeys], obdf['obs'], cv=kf, params=fitkwds
        )
        regr.fit(obdf[xkeys], obdf['obs'], **fitkwds)
        tgtdf['mod_bc'] = regr.predict(tgtX)

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
        os.makedirs(os.path.dirname(ncpath), exist_ok=True)
        tgtds.to_netcdf(ncpath)

        ds = xr.open_dataset(ncpath)
        to_geojson(
            jpath, x=ds.x, y=ds.y, z=ds['mod_bc'][0], crs=ds.crs_proj4,
            edges=_edges, colors=_colors, under=_colors[0], over=_colors[-1],
            description=ds.description
        )
        if _debug:
            os.system(f'cp {jpath} ./')
            os.system(f'cp {ncpath} ./')
