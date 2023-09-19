__all__ = ['fuse']

from .naqfc import get_mostrecent as get_naqfc
from .geoscf import get_mostrecent as get_geoscf
from .obs import pair_airnow, pair_purpleair
from .models import applymodel, get_models
import time
import pyproj
import os
import logging
import pandas as pd


def fuse(
    obssource, species, startdate, model, bbox=None, cv_only=False,
    outdir=None, overwrite=False, api_key=None, **kwds
):
    """
    Arguments
    ---------
    species : str
    startdate : datelike
    model : str
    bbox : list
    cv_only : bool
    outdir : str or None
    overwrite : bool
    obssource : str

    Returns
    -------
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
    logpath = f'{stem}.log'
    cvpath = f'{stem}_CV.csv'
    fusepath = f'{stem}.csv'

    found = set()
    for path in [cvpath, fusepath]:
        if os.path.exists(path):
            found.add(path)

    if len(found) > 0 and not overwrite:
        raise IOError(f'Outputs exist; delete or use -O to continue:\n{found}')

    logging.basicConfig(filename=logpath, level=logging.INFO)

    if model == 'NAQFC':
        noaakey = {'pm25': 'LZQZ99_KWBP', 'o3': 'LYUZ99_KWBP'}[species]
        modvar = get_naqfc(date, key=noaakey)
    elif model == 'GEOSCF':
        nasakey = {'pm25': 'pm25_rh35_gcc', 'o3': 'o3'}[species]
        modvar = get_geoscf(date, key=nasakey, bbox=bbox)
    else:
        raise KeyError(f'{model} unknown')

    modvar.name = model
    proj = pyproj.Proj(modvar.attrs['crs_proj4'], preserve_units=True)
    logging.info(proj.srs)

    obskey = {'o3': 'ozone', 'pm25': 'pm25'}[species]
    if obssource == 'airnow':
        obsdf = pair_airnow(date, bbox, proj, modvar, obskey)
    elif obssource == 'purpleair':
        obsdf = pair_purpleair(
            date, bbox, proj, modvar, obskey, api_key=api_key
        )

    models = get_models()

    if cv_only:
        tgtdf = None
    else:
        outdf = modvar.to_dataframe().reset_index()
        tgtdf = outdf.query(f'{modvar.name} == {modvar.name}').copy()

    # Apply all models to observations
    for mkey, mod in models.items():
        logging.info(f'{obssource} {mkey} start')
        t0 = time.time()
        applymodel(
            mod, mkey, obsdf, tgtdf=tgtdf, obskey=obskey, modkey=modvar.name,
            verbose=9
        )
        t1 = time.time()
        logging.info(f'{obssource} {mkey} finish: {t1 - t0:.0f}s')

    # Save results to disk
    obsdf.to_csv(cvpath, index=False)
    if not cv_only:
        tgtdf.to_csv(fusepath, index=False)

    logging.info('Successful completion')
    return {'outpath': fusepath, 'evalpath': cvpath, 'logpath': logpath}


if __name__ == '__main__':
    from .parser import parse_args
    args = parse_args()
    fuse(**vars(args))
