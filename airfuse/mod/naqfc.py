__all__ = ['open_mostrecent', 'get_mostrecent']


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


def open_mostrecent(date, key='LZQZ99_KWBP', failback='24h', filedate=None):
    """
    Arguments
    ---------
    date : datetime-like
        Beginning hour of date to extract
    key : str
        LZQZ99_KWBP (pm) or LYUZ99_KWBP (ozone)
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    failback : str
        If file could not be found, find the previous XXh file.
    filedate : datetime-like
        Date of the file

    Returns
    -------
    var : xr.Dataset
        Dataset from NCEI archive of National Guidance Data Center
    """
    import xarray as xr
    import pandas as pd

    if filedate is None:
        filedate = date
    paths = getpaths(filedate, service='dodsC', key=key)
    if len(paths) == 0:
        raise IOError('Could not find relevant file.')
    for path in paths[::-1]:
        try:
            naqfcf = xr.open_dataset(path)
            naqfcf = naqfcf.sel(time=date, sigma=1)
            addcrs(naqfcf)
            return naqfcf
        except KeyError as e:
            last_err = e
            print(f'{date} not in {path}; testing next available file')
    else:
        if failback is not None:
            return open_mostrecent(
                date=date, key=key, failback=None,
                filedate=pd.to_datetime(date) - pd.to_timedelta(failback),
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
        LZQZ99_KWBP (pm) or LYUZ99_KWBP (ozone)

    filedate : datetime-like or None
        Date of file to open
    verbose : int
        Level of verbosity.
    source : str
        Source either 'nws' or 'ncep'.
        * 'nws' is the true operational site.
        * 'ncep' provides more thorough file naming.

    Results
    -------
    outf : xarray.Dataset
        Outputs a file that looks like the NCEI archive file opened as a
        NetCDF file. In addition, it will have a crs_proj4 attrribute that
        describes the projection of the underlying file.

    """
    import requests
    import xarray as xr
    import pandas as pd
    import numpy as np
    import tempfile
    import pyproj

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

    nws_cf227_proj4 = (
        '+proj=lcc +lat_1=25 +lat_0=25 +lon_0=265 +k_0=1 +x_0=0 +y_0=0'
        + ' +R=6371229 +units=km +no_defs'
    )
    nws_cf227_proj = pyproj.Proj(nws_cf227_proj4)
    pattrs = nws_cf227_proj.crs.to_cf()
    bdate = pd.to_datetime(bdate)
    if filedate is None:
        filedate = bdate.floor('1d')

    for sh in [18, 12, 6, 0]:
        firsth = filedate + pd.to_timedelta(sh + 1, unit='H')
        if firsth > bdate:
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
            print(url)
        try:
            r = requests.get(url)
            if r.status_code != 200:
                if verbose > 0:
                    print(f'Code {r.status_code} {url}')
                continue
            with tempfile.NamedTemporaryFile() as tf:
                tf.write(r.content)
                f = xr.open_dataset(tf.name, engine='cfgrib')
                f = f.drop_vars(['latitude', 'longitude'])
                # Minimum x and y are taken from an opened file from NCEI
                # archive. Units are in km
                minx = -4226.10839844
                miny = -832.69781494
                f.coords['x'] = np.arange(f.dims['x']) * 5.079 + minx
                f.coords['y'] = np.arange(f.dims['y']) * 5.079 + miny
                f['LambertConformal_Projection'] = xr.DataArray(
                    0, name='LambertConformal_Projection', dims=(),
                    attrs={k: v for k, v in pattrs.items()}
                )
                renames = dict(time='reftime', step='time')
                renames[oldkey] = varkey
                outf = f.drop('valid_time').rename(**renames)
                outf.coords['time_bounds'] = xr.DataArray(
                    np.append(f['time'].values, f['valid_time'].values),
                    name='time_bounds', dims=('time_bounds',)
                )
                outf.coords['time'] = xr.DataArray(
                    f.valid_time.values, name='time', dims=('time',),
                    attrs=dict(bounds='time_bounds')
                )
                outf.attrs['crs_proj4'] = nws_cf227_proj4
                return outf.sel(time=bdate).load()
        except requests.models.HTTPError:
            pass
        except Exception as e:
            raise e
    else:
        filedate = filedate - pd.to_timedelta(failback)
        return open_operational(
            bdate, filedate=filedate, verbose=verbose, source=source, key=key,
            failback=failback
        )


def get_mostrecent(date, key='LZQZ99_KWBP', failback='24h', path=None):
    """
    Arguments
    ---------
    date : datetime-like
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

    naqfcf = None
    if path is not None:
        if os.path.exists(path):
            naqfcf = xr.open_dataset(path)

    if naqfcf is None:
        date = pd.to_datetime(date, utc=True)
        dt = (pd.to_datetime('now', utc=True).floor('1d') - date.floor('1d'))
        ds = dt.total_seconds()
        if ds < (2 * 24 * 3600):
            naqfcf = open_operational(date, key=key, failback='24h')
        else:
            naqfcf = open_mostrecent(date, key=key, failback='24h')
        if path is not None:
            naqfcf.to_netcdf(path)

    if key.startswith('LZQZ99'):
        varkey = 'Particulate_matter_fine_sigma_1_Hour_Average'
    elif key.startswith('LYUZ99'):
        varkey = 'Ozone_Concentration_sigma_1_Hour_Average'
    else:
        raise KeyError(f'{key} unknown try LZQZ99_KWBP or LYUZ99_KWBP.')

    var = naqfcf[varkey].load()
    var.attrs['crs_proj4'] = naqfcf.attrs['crs_proj4']
    var.attrs['long_name'] = varkey
    var.name = 'NAQFC'
    return var
