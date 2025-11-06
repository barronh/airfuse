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
    from ..layers import naqfc
    from ..points import rsigairnow
    from ..utils import fuse, df2ds, to_geojson
    from .. import dnr

    with tempfile.TemporaryDirectory() as td:
        mod = naqfc(spc, nowcast=nowcast, inroot=td)
        # get acquires a local cache for reuse in nowcast
        modvar = mod.get(date)

        # Get Obs: AirNow & PurpleAir
        xkeys = ['x', 'y']
        ykeys = ['obs', 'mod']
        obdf = rsigairnow(spc, nowcast=nowcast, inroot=td).pair(
            date, modvar, mod.proj
        )[xkeys + ykeys]
        obdf['groups'] = 'an'
        obdf['sample_weight'] = 1.

        # Perform Fusion Using Grouped DNR
        # - Calculate one surface from pooled weights
        # - Weights calculated separately for groups
        #   - two base functions,
        #   - two Delaunay diagrams and functions
        yhatsfx = '_angdnr'
        cvsfx = f'{yhatsfx}_cv'
        ncpath = f'{td}/{date:%Y/%m/%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.nc'
        gdnr = dnr.GroupedDelaunayNeighborsRegressor(
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
        fuse(
            tgtdf, obdf, obdnr=gdnr, fitkwds=fitkwds, yhatsfx=yhatsfx,
            cvsfx=cvsfx
        )
        tgtdf.set_index('time', append=True, inplace=True)
        tgtds = df2ds(
            tgtdf, obdf, yhatsfx=yhatsfx, cvsfx=cvsfx,
            crs_proj4=modvar.crs_proj4
        )
        os.makedirs(os.path.dirname(ncpath), exist_ok=True)
        tgtds.to_netcdf(ncpath)
        ds = xr.open_dataset(ncpath)
        jpath = f'{td}/{date:%Y/%m/%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.geojson'
        to_geojson(
            jpath, x=ds.x, y=ds.y, z=ds['bc_gdnr'][0], crs=ds.crs_proj4,
            edges=_edges, colors=_colors, under=_colors[0], over=_colors[-1],
            description=ds.description
        )
        if _debug:
            os.system(f'cp {jpath} ./')
            os.system(f'cp {ncpath} ./')


@pytest.mark.skipif(not _haspakey, reason="requires ~/.purpleairkey")
def test_fullexample():
    from ..layers import naqfc
    from ..points import rsigairnow, rsigpurpleair
    from ..utils import fuse, df2ds, to_geojson
    from .. import dnr
    with tempfile.TemporaryDirectory() as td:
        mod = naqfc(spc, nowcast=nowcast, inroot=td)
        # get acquires a local cache for reuse in nowcast
        modvar = mod.get(date)

        # Get Obs: AirNow & PurpleAir
        andf = rsigairnow(spc, nowcast=nowcast, inroot=td).pair(
            date, modvar, mod.proj
        )
        padf = rsigpurpleair(spc, nowcast=nowcast, inroot=td).pair(
            date, modvar, mod.proj
        )
        xkeys = ['x', 'y']
        ykeys = ['obs', 'mod']
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
        yhatsfx = '_gdnr'
        cvsfx = f'{yhatsfx}_cv'
        ncpath = f'{td}/{date:%Y/%m/%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.nc'
        gdnr = dnr.GroupedDelaunayNeighborsRegressor(
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
        fuse(
            tgtdf, obdf, obdnr=gdnr, fitkwds=fitkwds, yhatsfx=yhatsfx,
            cvsfx=cvsfx
        )
        tgtdf.set_index('time', append=True, inplace=True)
        tgtds = df2ds(
            tgtdf, obdf, yhatsfx=yhatsfx, cvsfx=cvsfx,
            crs_proj4=modvar.crs_proj4
        )
        os.makedirs(os.path.dirname(ncpath), exist_ok=True)
        tgtds.to_netcdf(ncpath)
        ds = xr.open_dataset(ncpath)
        jpath = f'{td}/{date:%Y/%m/%d/AirFuse.%Y-%m-%dT%H}Z{yhatsfx}.geojson'
        to_geojson(
            jpath, x=ds.x, y=ds.y, z=ds['bc_gdnr'][0], crs=ds.crs_proj4,
            edges=_edges, colors=_colors, under=_colors[0], over=_colors[-1],
            description=ds.description
        )
        if _debug:
            os.system(f'cp {jpath} ./')
            os.system(f'cp {ncpath} ./')
