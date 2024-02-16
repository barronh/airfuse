__all__ = ['fuse']

from .mod import get_model
from .obs import pair_airnow, pair_aqs, pair_purpleair
from .models import applyfusion, get_fusions
from .util import df2nc
import time
import pyproj
import os
import logging
import pandas as pd
from . import __version__
import warnings


def fuse(
    obssource, species, startdate, model, bbox=None, cv_only=False,
    outdir=None, overwrite=False, api_key=None, verbose=0, **kwds
):
    """
    Arguments
    ---------
    obssource : str
    species : str
    startdate : datelike
    model : str
    bbox : list
    cv_only : bool
    outdir : str or None
    overwrite : bool

    Returns
    -------
    outpaths : dict
        Dictionary of output paths
    """
    date = pd.to_datetime(startdate)
    if obssource == 'purpleair' and species != 'pm25':
        raise KeyError(
            f'{obssource} only available with pm25; you chose {species}'
        )
    if bbox is None:
        bbox = (-135, 15, -55, 60)
    model = model.upper()
    spctitle = species.upper()
    if outdir is None:
        outdir = f'{date:%Y/%m/%d}'

    os.makedirs(outdir, exist_ok=True)
    stem = (
        f'{outdir}/Fusion_{spctitle}_{model}_{obssource}_{date:%Y-%m-%dT%H}Z'
    )
    outfmt = kwds.get("format", 'csv')
    logpath = f'{stem}.log'
    cvpath = f'{stem}_CV.csv'
    fusepath = f'{stem}.{outfmt}'
    outpaths = {'outpath': fusepath, 'evalpath': cvpath, 'logpath': logpath}

    found = {k: os.path.exists(p) for k, p in outpaths.items()}
    chks = ['evalpath', 'outpath']

    if any([found[k] for k in chks]) and not overwrite:
        foundstr = ' '.join([outpaths[k] for k, v in found.items() if v])
        warnings.warn(
            f'Outputs exist; delete or use -O to continue:\n{foundstr}'
        )
        return outpaths

    logging.basicConfig(filename=logpath, level=logging.INFO)
    logging.info(f'AirFuse {__version__}')
    logging.info(f'Output dir: {outdir}')

    modvar = get_model(
        date, key=species, bbox=bbox, model=model, verbose=verbose
    )
    logging.info(f'Model: {modvar.description}')
    proj = pyproj.Proj(modvar.attrs['crs_proj4'], preserve_units=True)
    logging.info(proj.srs)

    obskey = {'o3': 'ozone', 'pm25': 'pm25'}[species]
    if obssource == 'airnow':
        obsdf = pair_airnow(date, bbox, proj, modvar, obskey)
    elif obssource == 'aqs':
        obsdf = pair_aqs(date, bbox, proj, modvar, obskey)
    elif obssource == 'purpleair':
        obsdf = pair_purpleair(
            date, bbox, proj, modvar, obskey, api_key=api_key
        )
    logging.info(f'{obssource} N={obsdf.shape[0]}')
    vardescs = {
      'NAQFC': 'NOAA Forecast (NAQFC)',
      f'IDW_{obskey}': f'NN weighted (n=10, d**-5) AirNow {obskey}',
      f'VNA_{obskey}': f'VN weighted (n=nv, d**-2) AirNow {obskey}',
      'aIDW': 'IDW of AirNow bias added to the NOAA NAQFC forecast',
      'aVNA': 'VNA of AirNow bias added to the NOAA NAQFC forecast',
    }
    varattrs = {
        k: dict(description=v, units='micrograms/m**3')
        for k, v in vardescs.items()
    }
    nowstr = pd.to_datetime('now', utc=True).strftime('%Y-%m-%dT%H:%M:%S%z')
    fdesc = f"""Fusion of observations (AirNow and PurpleAir) using residual
interpolation and correction of the NOAA NAQFC forecast model. The bias is
estimated in real-time using AirNow and PurpleAir measurements. It is
interpolated using the average of either nearest neighbors (IDW) or the
Voronoi/Delaunay neighbors (VNA). IDW uses 10 nearest neighbors with a
weight equal to distance to the -5 power. VNA uses just the Delaunay
neighbors and a weight equal to distnace to the -2 power. The aVNA and aIDW
use an additive bias correction using these interpolations.

{model}: {modvar.description}
{obssource} N=: {obsdf.shape[0]}
updated: {nowstr}
"""

    models = get_fusions()

    if cv_only:
        tgtdf = None
    else:
        outdf = modvar.to_dataframe().reset_index()
        tgtdf = outdf.query(f'{modvar.name} == {modvar.name}').copy()

    # Apply all models to observations
    for mkey, mod in models.items():
        logging.info(f'{obssource} {mkey} start')
        t0 = time.time()
        applyfusion(
            mod, mkey, obsdf, tgtdf=tgtdf, obskey=obskey, modkey=modvar.name,
            verbose=9
        )
        t1 = time.time()
        logging.info(f'{obssource} {mkey} finish: {t1 - t0:.0f}s')

    # Save results to disk
    obsdf.to_csv(cvpath, index=False)
    if not cv_only:
        # Save final results to disk
        if fusepath.endswith('.nc'):
            metarow = tgtdf.iloc[0]
            fileattrs = {
                'title': f'AirFuse ({__version__}) {obskey}',
                'author': 'Barron H. Henderson',
                'institution': 'US Environmental Protection Agency',
                'description': fdesc, 'crs_proj4': proj.srs,
                'reftime': metarow['reftime'].strftime('%Y-%m-%dT%H:%M:%S%z'),
                'sigma': metarow['sigma'],
                'updated': nowstr
            }
            tgtds = df2nc(tgtdf, varattrs, fileattrs)
            tgtds.to_netcdf(fusepath)
        else:
            # Defualt to csv
            tgtdf.to_csv(fusepath, index=False)
    logging.info('Successful completion')
    if cv_only:
        outpaths.pop('outpath', 0)

    return outpaths


if __name__ == '__main__':
    from .parser import parse_args
    args = parse_args()
    fuse(**vars(args))
