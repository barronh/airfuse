__all__ = ['get_mostrecent']

import logging
logger = logging.getLogger(__name__)


def get_mostrecent(
    bdate, key='o3', bbox=None, failback='24h', resfac=4, filedate=None,
    path=None
):
    """
    Arguments
    ---------
    bdate : datetime-like
        Beginning hour of date to extract
    key : str
        pm25_rh35_gcc, o3, no2, so2, or co
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    failback : str
        If file could not be found, find the previous XXh file.
    resfac : int
        Factor to increase the resolution by via interpolation
    filedate : datetime-like
        Date of the file
    path : str or None
        Path to archive result for reuse

    Returns
    -------
    var : xr.DataArray
        DataArray with values at cell centers and a projection stored as
        the attribute crs_proj4
    """
    import xarray as xr
    import numpy as np
    import pandas as pd
    import warnings

    fcast = 'https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/fcast'
    froot = 'aqc_tavg_1hr_g1440x721_v1/aqc_tavg_1hr_g1440x721_v1'
    bdate = pd.to_datetime(bdate)
    mdate = bdate + pd.to_timedelta('30min')
    if key == 'pm25':
        key = 'pm25_rh35_gcc'

    if filedate is None:
        filedate = bdate
        if bdate.hour < 12:
            filedate = filedate + pd.to_timedelta('-24h')
    else:
        filedate = pd.to_datetime(filedate)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = xr.open_dataset(
                f'{fcast}/{froot}.{filedate:%Y%m%d}_12z'
            )
        var = f[key].sel(time=mdate, method='nearest')
    except Exception as e:
        logger.info(str(e))
        filedate = filedate - pd.to_timedelta(failback)
        return get_mostrecent(
            bdate, key=key, bbox=bbox, failback=failback, filedate=filedate
        )
    if bbox is not None:
        var = var.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[1], bbox[3]))
    if key == 'o3':
        var = (var * 1e9)
        var.attrs['units'] = 'ppb'
    if key == 'pm25_rh35_gcc':
        var.attrs['units'] = 'micrograms/m**3 at 35RH'
        var.attrs['description'] = var.attrs['long_name']
        var.attrs['long_name'] = 'pm25'
    var.attrs['crs_proj4'] = '+proj=lonlat +ellps=WGS84 +no_defs'
    var = var.load().rename(lon='x', lat='y').sel(lev=72)
    if resfac != 1:
        # Increase the spatial resolution by a factor of
        xi = np.linspace(var.x.min(), var.x.max(), (var.x.size - 1) * 4 + 1)
        yi = np.linspace(var.y.min(), var.y.max(), (var.y.size - 1) * 4 + 1)
        var = var.interp(x=xi, y=yi)
    return var
