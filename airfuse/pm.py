__all__ = ['pmfuse']

from .mod import get_model
from .obs import pair
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
import requests
import pandas as pd


def pmfuse(
    startdate, model, bbox=None, dust_ev_filt=False, cv_only=False,
    outdir=None, overwrite=False, api_key=None, verbose=0, njobs=None,
    modvar=None, andf=None, padf=None, exclude_stations=True, format='csv',
    **kwds
):
    """
    Must accept all arguments from airfuse.parser.parse_args

    Arguments
    ---------
    startdate : datelike
        Beginning hour of prediction
    model : str
        Currently works with naqfc, geoscf, goes
    bbox : list
        wlon, slat, elon, nlat in decimal degrees east and north
        lon >= -180 and lon <= 180
        lat >= -90 and lat <= 90
    dust_ev_filt: bool
        If True, this removes sites where a dust event is likely
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
    andf : pandas.DataFrame
        Optional, used to bypass typical AirNow acquisition process.
        Must have pm25 column, x/y columns consistent with modvar, and a NAQFC
        or GEOSCF column from modvar
    padf : pandas.DataFrame
        Optional, used to bypass typical AirNow acquisition process.
        Must have pm25 column, x/y columns consistent with modvar, and a NAQFC
        or GEOSCF column from modvar
    exclude_stations : bool
        If True, use excluded stations identified by the Fire And Smoke Map.
        This exclusion is only relevant if fusion target hour is within 1d of
        now.
    format : str
        Default csv (easily readable by anyone); also supports netcdf (nc)
    kwds : mappable
        Other unknown keywords

    Returns
    -------
    outpaths : dict
        Dictionary of output paths
    """
    date = pd.to_datetime(startdate)
    model = model.upper()
    obskey = 'pm25'

    vardescs = {
      'NAQFC': 'NOAA Forecast (NAQFC)',
      'IDW_AN_DIST': 'Distance to AirNow [km]',
      'IDW_PA_DIST': 'Distance to PurpleAir [km]',
      f'IDW_AN_{obskey}': f'NN weighted (n=10, d**-5) AirNow {obskey}',
      f'VNA_AN_{obskey}': f'VN weighted (n=nv, d**-2) AirNow {obskey}',
      'aIDW_AN': 'IDW of AirNow bias added to the NOAA NAQFC forecast',
      'aVNA_AN': 'VNA of AirNow bias added to the NOAA NAQFC forecast',
      'FUSED_aVNA': 'Fused surface from aVNA PurpleAir and aVNA AirNow',
      'FUSED_eVNA': 'Fused surface from eVNA PurpleAir and eVNA AirNow',
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
    outfmt = format
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

    if modvar is None:
        modvar = get_model(
            date, key=obskey, bbox=bbox, model=model, verbose=verbose
        )
    logging.info(f'{model}: {modvar.description}')
    fdesc = '\n'.join([fdesc, f'{model}: {modvar.description}'])
    # When merging fused surfaces, PurpleAir is treated as never being closer
    # than half the diagonal distance. Thsi ensures that AirNow will be the
    # preferred estimate within a grid cell if it exists. This is particularly
    # reasonable given that the PA coordinates are averaged
    dx = np.diff(modvar.x).mean()
    dy = np.diff(modvar.y).mean()
    pamindist = ((dx**2 + dy**2)**.5) / 2

    proj = pyproj.Proj(modvar.attrs['crs_proj4'], preserve_units=True)
    logging.info(proj.srs)

    if andf is None:
        andf = pair(date, bbox, proj, modvar, obskey, 'airnow')
    logging.info(f'AirNow N={andf.shape[0]}')
    fdesc = '\n'.join([fdesc, f'AirNow N={andf.shape[0]}'])

    if exclude_stations is True:
        if (pd.to_datetime('now', utc=True) - date).total_seconds() > (86400):
            exclude_stations = []

    if exclude_stations is True:
        paexcludeurl = (
            'https://airfire-data-exports.s3.us-west-2.amazonaws.com/elwood/'
            + 'exclusion_lists/elwood_exclusion.json'
        )
        fasm_exclude = requests.get(paexcludeurl, stream=True).json()
        exclude_stations = []
        for rec in fasm_exclude:
            if pd.to_datetime(rec['added']) <= date:
                exclude_stations.append(rec['unit_id'])
        logging.info(
            'Using FASM PurpleAir Exclusions: '
            + ''.join(exclude_stations)
        )

    if padf is None:
        padf = pair(
            date, bbox, proj, modvar, obskey, 'purpleair', api_key=api_key,
            dust_ev_filt=dust_ev_filt, exclude_stations=exclude_stations
        )

    logging.info(f'PurpleAir N={padf.shape[0]}')
    fdesc = '\n'.join([fdesc, f'PurpleAir N={padf.shape[0]}'])
    ann = andf.shape[0]
    pan = padf.shape[0]
    minv = min([30, int(ann * .9), int(pan * .9)])
    mini = min([10, int(ann * .9), int(pan * .9)])
    if ann < 30:
        logging.info(f'AirNow had only {ann} obs')
    if pan < 30:
        logging.info(f'PurpleAir had only {pan} obs')
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

    # Apply all models to AirNow observations
    for mkey, mod in models.items():
        logging.info(f'AN {mkey} begin')
        t0 = time.time()
        applyfusion(
            mod, f'{mkey}_AN', andf, tgtdf=tgtdf, obskey=obskey,
            modkey=modvar.name, verbose=9, njobs=njobs
        )
        t1 = time.time()
        logging.info(f'AN {mkey} {t1 - t0:.0f}s')

    # Apply all models to PurpleAir observations
    for mkey, mod in models.items():
        logging.info(f'PA {mkey} begin')
        t0 = time.time()
        applyfusion(
            mod, f'{mkey}_PA', padf, tgtdf=tgtdf, loodf=andf,
            obskey=obskey, modkey=modvar.name, verbose=9, njobs=njobs
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
            }
            if 'reftime' in metarow:
                fileattrs['reftime'] = (
                    metarow['reftime'].strftime('%Y-%m-%dT%H:%M:%S%z')
                )
            if 'sigma' in metarow:
                fileattrs['sigma'] = metarow['sigma']
            tgtds = df2nc(tgtdf, varattrs, fileattrs)
            andf['GRIDDED_FUSED_aVNA'] = tgtds['FUSED_aVNA'].sel(
                x=andf['x'].to_xarray(), y=andf['y'].to_xarray(),
                method='nearest'
            ).values.squeeze()
            tgtds.to_netcdf(fusepath)
        else:
            # Defualt to csv
            tgtdf.to_csv(fusepath, index=False)

    # Save results to disk as CSV files
    andf.to_csv(ancvpath, index=False)
    padf.to_csv(pacvpath, index=False)

    logging.info('Successful Completion')
    if cv_only:
        outpaths.pop('outpath', 0)

    return outpaths


if __name__ == '__main__':
    from .parser import parse_args
    args = parse_args()
    pmfuse(**vars(args))
