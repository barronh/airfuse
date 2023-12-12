__all__ = ['open_mostrecent', 'get_mostrecent', 'open_operational']

import logging
logger = logging.getLogger(__name__)


def getpaths(date, key, service='fileServer'):
    """
    Arguments
    ---------
    date : datetime-like
        Str or datetime or pandas.Datetime
    service : str
      'dodsC' or 'fileServer'

    Returns
    -------
    paths : list
        List of paths to files on the NAQFC server.
    """
    import xml.etree.ElementTree
    import requests
    import pandas as pd

    pdate = pd.to_datetime(date)
    r = requests.get(
      'https://www.ncei.noaa.gov/thredds/catalog/model-ndgd-file/'
      + f'{pdate:%Y%m/%Y%m%d}/catalog.xml'
    )
    et = xml.etree.ElementTree.fromstring(r.text)
    datasets = []
    for p in et:
        for c in p:
            if c.tag.endswith('dataset'):
                url = c.attrib.get('urlPath', 'none')
                if key in url:
                    datasets.append(url)
    return [
        f'https://www.ncei.noaa.gov/thredds/{service}/' + p
        for p in sorted(datasets)
    ]


def addcrs(naqfcf):
    """
    Adds projection to naqfcf

    Arguments
    ---------
    naqfcf : xr.Dataset
        Must have a variable named LambertConformal_Projection whose
        attributes describe the Climate Forecasting Conventions definition
        of the projection
    Returns
    -------
    None
    """
    import pyproj

    # Build CRS in m from CF convention mapping variable attributes
    crs = pyproj.CRS.from_cf(naqfcf['LambertConformal_Projection'].attrs)
    # Default projection is in m, but coordinate data is in km
    proj = pyproj.Proj(crs)
    # Use km units (consistent with x/y)
    naqfcf.attrs['crs_proj4'] = proj.srs.replace('+units=m', '+to_meter=1000')


def open_mostrecent(
    bdate, key='LZQZ99_KWBP', failback='24h', filedate=None, verbose=0
):
    """
    Arguments
    ---------
    bdate : datetime-like
        Beginning hour of date to extract
    key : str
        Hourly average CONUS: LZQZ99_KWBP (pm) or LYUZ99_KWBP (ozone)
        w/ bias correction: LOPZ99_KWBP (pm) or YBPZ99_KWBP (ozone)
        For more key options, see:
        https://www.nws.noaa.gov/directives/sym/pd01005016curr.pdf
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    failback : str
        If file could not be found, find the previous XXh file.
    filedate : datetime-like
        Date of the file
    verbose : int
        Level of verbosity

    Returns
    -------
    var : xr.Dataset
        Dataset from NCEI archive of National Guidance Data Center
    """
    import xarray as xr
    import pandas as pd
    edate = pd.to_datetime(bdate) + pd.to_timedelta('1H')
    if filedate is None:
        filedate = bdate
    paths = getpaths(filedate, service='dodsC', key=key)
    if len(paths) == 0:
        raise IOError('Could not find relevant file.')
    for path in paths[::-1]:
        try:
            naqfcf = xr.open_dataset(path)
            naqfcf = naqfcf.sel(time=edate, sigma=1)
            # Move "time" to midpoint, which helps prevent ambigous start/end
            naqfcf.coords['time'] = (
                naqfcf.coords['time'] + pd.to_timedelta('-30min')
            )
            addcrs(naqfcf)
            return naqfcf
        except KeyError as e:
            last_err = e
            logger.info(f'{bdate} not in {path}; testing next available file')
    else:
        if failback is not None:
            return open_mostrecent(
                bdate=bdate, key=key, failback=None,
                filedate=pd.to_datetime(bdate) - pd.to_timedelta(failback),
            )
        else:
            raise last_err


def open_operational(
    bdate, key='LZQZ99_KWBP', filedate=None, source='ncep', failback='24h',
    verbose=4
):
    """
    Arguments
    ---------
    bdate : datetime-like
        Beginning hour of hourly average
    key : str
        Hourly average CONUS: LZQZ99_KWBP (pm) or LYUZ99_KWBP (ozone)
        w/ bias correction: LOPZ99_KWBP (pm) or YBPZ99_KWBP (ozone)
        For more key options, see:
        https://www.nws.noaa.gov/directives/sym/pd01005016curr.pdf
    filedate : datetime-like or None
        Date of file to open
    verbose : int
        Level of verbosity.
    source : str
        Source either 'nws' or 'ncep' and only applies when requesting a file
        from the last two days.
        * 'nws' is the true operational site.
        * 'ncep' provides more thorough file naming.

    Results
    -------
    outf : xarray.Dataset
        Outputs a file that looks like the NCEI archive file opened as a
        NetCDF file. In addition, it will have a crs_proj4 attrribute that
        describes the projection of the underlying file.

    """
    import os
    import pandas as pd
    import requests
    import tempfile
    import xarray as xr
    import pyproj
    import numpy as np

    if key.startswith('LZQZ99'):
        oldkey = 'pmtf'
        varkey = 'Particulate_matter_fine_sigma_1_Hour_Average'
        nwscode = 'apm25h01'
        ncepcode = 'ave_1hr_pm25'
    elif key.startswith('LYUZ99'):
        oldkey = 'ozcon'
        varkey = 'Ozone_Concentration_sigma_1_Hour_Average'
        nwscode = 'ozone01'
        ncepcode = 'ave_1hr_o3'

    bdate = pd.to_datetime(bdate)
    edate = bdate + pd.to_timedelta('1H')
    if filedate is None:
        filedate = bdate.floor('1d')

    gridpath = f'{key}_GRID.nc'
    if not os.path.exists(gridpath):
        # this is an arbitrary NDGD file with a known grid
        # it's grid is consistent with KWBP CONUS runs
        expath = (
            'https://www.ncei.noaa.gov/thredds/dodsC/model-ndgd-file/'
            '202107/20210731/LZQZ99_KWBP_202107311100'
        )
        gridkeys = ['LambertConformal_Projection', 'y', 'x']
        gridds = xr.open_dataset(expath)[gridkeys].load()
        gridds.reset_coords('reftime', drop=True).to_netcdf(gridpath)
    gridds = xr.open_dataset(gridpath).load()

    # nws_cf227_proj4 = (
    #     '+proj=lcc +lat_1=25 +lat_0=25 +lon_0=265 +k_0=1 +x_0=0 +y_0=0'
    #     + ' +R=6371229 +units=km +no_defs'
    # )
    # nws_cf227_proj = pyproj.Proj(nws_cf227_proj4)
    # pattrs = nws_cf227_proj.crs.to_cf()
    pattrs = gridds['LambertConformal_Projection'].attrs
    proj = pyproj.Proj(pyproj.CRS.from_cf(pattrs))
    nws_cf227_proj4 = proj.srs.replace('units=m', 'units=km')
    # 0 and 18Z have only 6h... should these be used at all?
    for sh in [18, 12, 6, 0]:
        firsth = filedate + pd.to_timedelta(sh + 1, unit='H')
        lasth = filedate + pd.to_timedelta(
            {18: 6, 12: 72, 6: 72, 0: 6}[sh] + 1, unit='H'
        )
        if firsth > edate:
            continue
        if lasth < edate:
            continue
        # dt = bdate - filedate - pd.to_timedelta(sh, unit='H')
        # fh = round(dt.total_seconds() / 3600, 0)
        if source == 'ncep':
            url = (
                'https://ftp.ncep.noaa.gov/data/nccf/com/aqm/prod/'
                + f'cs.{filedate:%Y%m%d}/'
                + f'aqm.t{sh:02d}z.{ncepcode}.227.grib2'
            )
        elif source == 'nws':
            url = (
                'https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/'
                + f'DC.ndgd/GT.aq/AR.conus/ds.{nwscode}.bin'
            )
        if verbose > 0:
            logger.info(url)
        try:
            if verbose > 1:
                logger.info(f'URL: {url}')
            r = requests.get(url)
            if r.status_code != 200:
                if verbose > 0:
                    logger.info(f'Code {r.status_code} {url}')
                continue

            with tempfile.NamedTemporaryFile() as tf:
                tf.write(r.content)
                f = xr.open_dataset(tf.name, engine='cfgrib')
                f = f.drop_vars(['latitude', 'longitude'])
                # Coordinates are taken from NCEP NCEI OpenDAP
                # to ensure consistency. Units are in km
                f.coords['x'] = gridds['x']
                f.coords['y'] = gridds['y']
                lcc = gridds['LambertConformal_Projection']
                f['LambertConformal_Projection'] = lcc
                renames = dict(time='reftime', step='time')
                renames[oldkey] = varkey
                outf = f.drop('valid_time').rename(**renames)
                outf.coords['time_bounds'] = xr.DataArray(
                    np.append(f['time'].values, f['valid_time'].values),
                    name='time_bounds', dims=('time_bounds',)
                )
                # valid_time is the end of the hour
                outf.coords['time'] = xr.DataArray(
                    f.valid_time.values, name='time', dims=('time',),
                    attrs=dict(bounds='time_bounds')
                )
                outf.attrs['crs_proj4'] = nws_cf227_proj4
                outf = outf.sel(time=edate.replace(tzinfo=None)).load()
                # Set time to mid-point in hour to prevent ambiguous start/end
                outf.coords['time'] = (
                    outf.coords['time'] + pd.to_timedelta('-30min')
                )
                return outf
        except requests.models.HTTPError:
            continue
        except KeyError:
            # When 00 or 18Z are run, they only have 6 hours of data, which may
            # not include the file
            continue
        except Exception as e:
            raise e
    else:
        filedate = filedate - pd.to_timedelta(failback)
        return open_operational(
            bdate, filedate=filedate, verbose=verbose, source=source, key=key,
            failback=failback
        )


def get_mostrecent(
    bdate, key='LZQZ99_KWBP', bbox=None, failback='24h', path=None, verbose=0
):
    """
    Arguments
    ---------
    bdate : datetime-like
        Beginning hour of date to extract
    key : str
        LZQZ99_KWBP (pm25) or LYUZ99_KWBP (ozone)
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    failback : str
        If file could not be found, find the previous XXh file.
    filedate : datetime-like
        Date of the file
    path : str
        Path to archive result for reuse (not currently operational)
    verbose : int
        Level of verbosity

    Returns
    -------
    var : xr.DataArray
        DataArray with values at cell centers and a projection stored as
        the attribute crs_proj4

    To-do
    -----
    1. Update so that it can pull from live NWS feed
      https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndgd/GT.aq/
    2. Update so that it can pull from Hawaii or Alaska feeds
    3. Update so that it can pull from global feed
    """
    import os
    import xarray as xr
    import pandas as pd
    import pyproj

    naqfcf = None
    if path is not None:
        if os.path.exists(path):
            naqfcf = xr.open_dataset(path)

    if naqfcf is None:
        bdate = pd.to_datetime(bdate, utc=True)
        dt = (pd.to_datetime('now', utc=True).floor('1d') - bdate.floor('1d'))
        ds = dt.total_seconds()
        # Start date must be today (-0day) or yesterday (-1day)
        # if the result is older than -1day, use NCEI
        if ds < (1.5 * 24 * 3600):
            if verbose > 0:
                logger.info('Calling open_operational')
            naqfcf = open_operational(
                bdate, key=key, failback=failback, verbose=verbose
            )
        else:
            if verbose > 0:
                logger.info('Calling open_mostrecent')
            naqfcf = open_mostrecent(
                bdate.replace(tzinfo=None), key=key, failback=failback
            )
        if path is not None:
            naqfcf.to_netcdf(path)

    if key.startswith('LZQZ99'):
        varkey = 'Particulate_matter_fine_sigma_1_Hour_Average'
    elif key.startswith('LYUZ99'):
        varkey = 'Ozone_Concentration_sigma_1_Hour_Average'
    else:
        raise KeyError(f'{key} unknown try LZQZ99_KWBP or LYUZ99_KWBP.')

    var = naqfcf[varkey].load()
    if bbox is not None:
        # Find lon/lat coordinates of projected cell centroids
        proj = pyproj.Proj(naqfcf.attrs['crs_proj4'])
        Y, X = xr.broadcast(var.y, var.x)
        LON, LAT = proj(X.values, Y.values, inverse=True)
        LON = X * 0 + LON
        LAT = Y * 0 + LAT
        # Find projected box covering lon/lat box
        inlon = ((LON >= bbox[0]) & (LON <= bbox[2])).any('y')
        inlat = ((LAT >= bbox[1]) & (LAT <= bbox[3])).any('x')
        var = var.sel(x=inlon, y=inlat)

    var.attrs['crs_proj4'] = naqfcf.attrs['crs_proj4']
    var.attrs['long_name'] = varkey
    var.name = 'NAQFC'
    return var
