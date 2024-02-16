__all__ = ['pmfuse']

from .mod import get_model
from .obs import pair_airnow, pair_purpleair
from .models import applyfusion, get_fusions
from .ensemble import distweight
from .util import df2nc
import numpy as np
import time
import pyproj
import os
import logging
from . import __version__
import warnings
import pandas as pd


def pmfuse(
    startdate, model, bbox=None, cv_only=False,
    outdir=None, overwrite=False, api_key=None, verbose=0, **kwds
):

    date = pd.to_datetime(startdate)
    model = model.upper()
    obskey = 'pm25'

    vardescs = {
      'NAQFC': 'NOAA Forecast (NAQFC)',
      f'IDW_AN_{obskey}': f'NN weighted (n=10, d**-5) AirNow {obskey}',
      f'VNA_AN_{obskey}': f'VN weighted (n=nv, d**-2) AirNow {obskey}',
      'aIDW_AN': 'IDW of AirNow bias added to the NOAA NAQFC forecast',
      'aVNA_AN': 'VNA of AirNow bias added to the NOAA NAQFC forecast',
      'FUSED_aVNA': 'Fused surface from aVNA PurpleAir and aVNA AirNow',
      'FUSED_aIDW': 'Fused surface from aIDW PurpleAir and aIDW AirNow',
    }
    varattrs = {
        k: dict(description=v, units='micrograms/m**3')
        for k, v in vardescs.items()
    }

    fdesc = """Fusion of observations (AirNow and PurpleAir) using residual
interpolation and correction of the NOAA NAQFC forecast model. The bias is
estimated in real-time using AirNow and PurpleAir measurements. It is
interpolated using the average of either nearest neighbors (IDW) or the
Voronoi/Delaunay neighbors (VNA). IDW uses 10 nearest neighbors with a
weight equal to distance to the -5 power. VNA uses just the Delaunay
neighbors and a weight equal to distnace to the -2 power. The aVNA and aIDW
use an additive bias correction using these interpolations. Each algorithm
is applied to both AirNow monitors and PurpleAir low-cost sensors. The
"FUSED" surfaces combine both surfaces using weights based on distance to
nearest obs.
"""

    # edit outdir to change destination (e.g., %Y%m%d instead of %Y/%m/%d)
    outdir = f'{date:%Y/%m/%d}'
    os.makedirs(outdir, exist_ok=True)
    stem = f'{outdir}/Fusion_PM25_{model}_{date:%Y-%m-%dT%H}Z'
    logpath = f'{stem}.log'
    pacvpath = f'{stem}_PurpleAir_CV.csv'
    ancvpath = f'{stem}_AirNow_CV.csv'
    outfmt = kwds.get("format", 'csv')
    fusepath = f'{stem}.{outfmt}'
    outpaths = {
        'outpath': fusepath, 'paevalpath': pacvpath, 'anevalpath': ancvpath,
        'logpath': logpath
    }

    found = {k: os.path.exists(p) for k, p in outpaths.items()}
    chks = ['anevalpath', 'paevalpath', 'outpath']

    if any([found[k] for k in chks]) and not overwrite:
        foundstr = ' '.join([outpaths[k] for k, v in found.items() if v])
        warnings.warn(
            f'Outputs exist; delete or use -O to continue:\n{foundstr}'
        )
        return outpaths

    # Divert all logging during this script to the associated
    # log file at the INFO level.
    logging.basicConfig(filename=logpath, level=logging.INFO)
    logging.info(f'AirFuse {__version__}')
    logging.info(f'Output dir: {outdir}')

    pm = get_model(date, key=obskey, bbox=bbox, model=model, verbose=verbose)
    logging.info(f'{model}: {pm.description}')
    fdesc = '\n'.join([fdesc, f'{model}: {pm.description}'])
    # When merging fused surfaces, PurpleAir is treated as never being closer
    # than half the diagonal distance. Thsi ensures that AirNow will be the
    # preferred estimate within a grid cell if it exists. This is particularly
    # reasonable given that the PA coordinates are averaged
    dx = np.diff(pm.x).mean()
    dy = np.diff(pm.y).mean()
    pamindist = ((dx**2 + dy**2)**.5) / 2

    proj = pyproj.Proj(pm.attrs['crs_proj4'], preserve_units=True)
    logging.info(proj.srs)

    andf = pair_airnow(date, bbox, proj, pm, obskey)
    logging.info(f'AirNow N={andf.shape[0]}')
    fdesc = '\n'.join([fdesc, f'AirNow N={andf.shape[0]}'])
    padf = pair_purpleair(date, bbox, proj, pm, obskey, api_key=api_key)
    logging.info(f'PurpleAir N={padf.shape[0]}')
    fdesc = '\n'.join([fdesc, f'PurpleAir N={padf.shape[0]}'])

    models = get_fusions()

    if cv_only:
        tgtdf = None
    else:
        outdf = pm.to_dataframe().reset_index()
        tgtdf = outdf.query(f'{pm.name} == {pm.name}').copy()

    # Apply all models to AirNow observations
    for mkey, mod in models.items():
        logging.info(f'AN {mkey} begin')
        t0 = time.time()
        applyfusion(
            mod, f'{mkey}_AN', andf, tgtdf=tgtdf, obskey=obskey,
            modkey=pm.name, verbose=9
        )
        t1 = time.time()
        logging.info(f'AN {mkey} {t1 - t0:.0f}s')

    # Apply all models to PurpleAir observations
    for mkey, mod in models.items():
        logging.info(f'PA {mkey} begin')
        t0 = time.time()
        applyfusion(
            mod, f'{mkey}_PA', padf, tgtdf=tgtdf, loodf=andf,
            obskey=obskey, modkey=pm.name, verbose=9
        )
        t1 = time.time()
        logging.info(f'PA {mkey} finish: {t1 - t0:.0f}s')

    # Force PA downweighting in same cell and neighboring cell.
    # Has no effect on LOO because nearest (ie, same cell) is already removed.
    loopaadjdist = np.maximum(andf['LOO_VNA_PA_DIST'], pamindist)
    andf['LOO_VNA_PA_DIST_ADJ'] = loopaadjdist
    # Perform fusions on LOO data for aVNA
    distkeys = ['LOO_VNA_AN_DIST', 'LOO_VNA_PA_DIST_ADJ']
    valkeys = ['LOO_aVNA_AN', 'LOO_aVNA_PA']
    distweight(
        andf, distkeys, valkeys, modkey=model, ykey='FUSED_aVNA', power=-2,
        add=True, LOO_aVNA_PA=0.25
    )
    # Perform fusions on LOO data for eVNA
    valkeys = ['LOO_eVNA_AN', 'LOO_eVNA_PA']
    distweight(
        andf, distkeys, valkeys, modkey=model, ykey='FUSED_eVNA', power=-2,
        add=True, LOO_eVNA_PA=0.25
    )
    # Perform fusions on LOO data for eVNA
    valkeys = ['LOO_aIDW_AN', 'LOO_aIDW_PA']
    distweight(
        andf, distkeys, valkeys, modkey=model, ykey='FUSED_aIDW', power=-2,
        add=True, LOO_aIDW_PA=0.25
    )
    # Save results to disk as CSV files
    andf.to_csv(ancvpath, index=False)
    padf.to_csv(pacvpath, index=False)

    if not cv_only:
        # Force PA downweighting in same cell and neighboring cell.
        tgtdf['VNA_PA_DIST_ADJ'] = np.maximum(tgtdf['VNA_PA_DIST'], pamindist)
        # Perform fusions on Target Dataset for aVNA
        distkeys = ['VNA_AN_DIST', 'VNA_PA_DIST_ADJ']
        valkeys = ['aVNA_AN', 'aVNA_PA']
        distweight(
            tgtdf, distkeys, valkeys, modkey=model, ykey='FUSED_aVNA',
            power=-2, add=True, aVNA_PA=0.25
        )
        # Perform fusions on Target Dataset for aIDW
        distkeys = ['VNA_AN_DIST', 'VNA_PA_DIST_ADJ']
        valkeys = ['aIDW_AN', 'aIDW_PA']
        distweight(
            tgtdf, distkeys, valkeys, modkey=model, ykey='FUSED_aIDW',
            power=-2, add=True, aIDW_PA=0.25
        )
        # Save final results to disk
        if fusepath.endswith('.nc'):
            metarow = tgtdf.iloc[0]
            fileattrs = {
                'title': f'AirFuse ({__version__}) {obskey}',
                'author': 'Barron H. Henderson',
                'institution': 'US Environmental Protection Agency',
                'description': fdesc, 'crs_proj4': proj.srs,
                'reftime': metarow['reftime'].strftime('%Y-%m-%dT%H:%M:%S%z'),
                'sigma': metarow['sigma']
            }
            tgtds = df2nc(tgtdf, varattrs, fileattrs)
            tgtds.to_netcdf(fusepath)
        else:
            # Defualt to csv
            tgtdf.to_csv(fusepath, index=False)

    logging.info('Successful Completion')
    if cv_only:
        outpaths.pop('outpath', 0)

    return outpaths


if __name__ == '__main__':
    from .parser import parse_args
    args = parse_args()
    pmfuse(**vars(args))
