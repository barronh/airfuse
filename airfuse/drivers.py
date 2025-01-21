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
    outdir=None, overwrite=False, api_key=None, verbose=0, njobs=None,
    modvar=None, obsdf=None, **kwds
):
    """
    Must accept all arguments from airfuse.parser.parse_args

    Arguments
    ---------
    obssource : str
        Data source to use for observations (airnow, purpleair, aqs)
    species : str
        Conceptually supports anything in airnow/aqs, but has been tested with
        o3 and pm25
    startdate : datelike
        Beginning hour of prediction
    model : str
        Currently works with naqfc, geoscf, goes
    bbox : list
        wlon, slat, elon, nlat in decimal degrees east and north
        lon >= -180 and lon <= 180
        lat >= -90 and lat <= 90
    cv_only : bool
        Only perform the cross-validation
    outdir : str or None
        Directory for output (Defaults to %Y/%m/%d)
    overwrite : bool
        If True, overwrite existing outpus.
    api_key : str or None
        Key for airnow or purpleair
    verbose : int
        Degree of verbosity
    njobs : int or None
        Number of processes to run during the target prediction phase.
    modvar : xarray.DataArray
        Optional, used to bypass typicall model acquisition process.
        Must have crs_proj4 and should be named NAQFC or GEOSCF.
    obsdf : pandas.DataFrame
        Optional, used to bypass typical observations acquisition process.
        Must have species column, x/y columns consistent with modvar, and a
        NAQFC or GEOSCF column from modvar

    kwds : mappable
        Other unknown keywords

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
    units = modvar.attrs.get('units', 'unknown').strip()
    logging.info(f'Model: {modvar.description}')
    proj = pyproj.Proj(modvar.attrs['crs_proj4'], preserve_units=True)
    logging.info(proj.srs)

    obskey = {'o3': 'ozone', 'pm25': 'pm25'}[species]
    if obsdf is None:
        if obssource == 'airnow':
            obsdf = pair_airnow(
                date, bbox, proj, modvar, obskey, api_key=api_key
            )
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
        k: dict(description=v, units=units)
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
    obsn = obsdf.shape[0]
    minv = min([30, int(obsn * .9)])
    mini = min([10, int(obsn * .9)])
    if obsn < 30:
        logging.info(f'Obs had only {obsn} obs')
    if minv < 30:
        logging.info(f'Only using nearest {minv} neighbors for voronoi')
    if mini < 10:
        logging.info(f'Only using nearest {mini} neighbors for IDW')
    models = get_fusions(v=minv, i=mini)

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
            verbose=9, njobs=njobs
        )
        t1 = time.time()
        logging.info(f'{obssource} {mkey} finish: {t1 - t0:.0f}s')

    if not cv_only:
        # Save final results to disk
        if fusepath.endswith('.nc'):
            metarow = tgtdf.iloc[0]
            fileattrs = {
                'title': f'AirFuse ({__version__}) {obskey}',
                'author': 'Barron H. Henderson',
                'institution': 'US Environmental Protection Agency',
                'description': fdesc, 'crs_proj4': proj.srs,
                'updated': nowstr
            }
            if 'reftime' in metarow:
                fileattrs['reftime'] = (
                    metarow['reftime'].strftime('%Y-%m-%dT%H:%M:%S%z')
                )
            if 'sigma' in metarow:
                fileattrs['sigma'] = metarow['sigma']

            tgtds = df2nc(tgtdf, varattrs, fileattrs)
            obsdf['GRIDDED_aVNA'] = tgtds['aVNA'].sel(
                x=obsdf['x'].to_xarray(),
                y=obsdf['y'].to_xarray(),
                method='nearest'
            ).values.squeeze()
            tgtds.to_netcdf(fusepath)
        else:
            # Defualt to csv
            tgtdf.to_csv(fusepath, index=False)

    # Save results to disk
    obsdf.to_csv(cvpath, index=False)

    logging.info('Successful completion')
    if cv_only:
        outpaths.pop('outpath', 0)

    return outpaths


if __name__ == '__main__':
    from .parser import parse_args
    args = parse_args()
    fuse(**vars(args))
