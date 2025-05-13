__all__ = ['open_mostrecent', 'get_mostrecent', 'open_operational']
__doc__ = """
NOAA Air Quality Forecast Capability (NAQFC)
--------------------------------------------

This module provides functionality for retrieving NAQFC and presenting a
format that AirFuse can work with.
"""
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
    import os

    pdate = pd.to_datetime(date)
    croot = 'https://www.ncei.noaa.gov/thredds/catalog/model-ndgd-file'
    # BHH: Undocumented feature to use the "historical" NCEI folder
    # The NCEI NDGD folder is missing data from 2019-12-16 to 2020-05-08.
    # (https://www.ncei.noaa.gov/thredds/catalog/model-ndgd-file/catalog.html)
    # The historical subfolder has data from 2019-12-01 to 2020-05-15.
    # (https://.../model-ndgd-file/historical/catalog.html)
    if os.environ.get('NDGD_HISTORICAL', 'F')[:1] in ('T', 'Y', 't', 'y'):
        croot = f'{croot}/historical'
    catalogurl = f'{croot}/{pdate:%Y%m/%Y%m%d}/catalog.xml'
    r = requests.get(catalogurl)
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


def getgrid(key='LZQZ99_KWBP'):
    """
    Arguments
    ---------
    key : str
        NCEP code for forecast (e.g., LZQZ99_KWBP)

    Returns
    -------
    gridds : xarray.Dataset
        Grid with x/y/LambertConformal_Projection either from NCEI or from an
        offline archive on github.
    """
    import os
    import xarray as xr
    import requests

    gridpath = f'{key}_GRID.nc'
    gridkey = key.split('_')[-1]
    assert gridkey == 'KWBP'
    if not os.path.exists(gridpath):
        # this is an arbitrary NDGD file with a known grid
        # it's grid is consistent with KWBP CONUS runs
        expath = (
            'https://www.ncei.noaa.gov/thredds/dodsC/model-ndgd-file/'
            '202107/20210731/LZQZ99_KWBP_202107311100'
        )
        gridkeys = ['LambertConformal_Projection', 'y', 'x']
        try:
            gridds = xr.open_dataset(expath)[gridkeys].load()
            gridds.attrs['file_url'] = expath
            gridds.reset_coords('reftime', drop=True).to_netcdf(gridpath)
        except Exception:
            # If there is an exception, it means that the file could not be
            # retrieved from the thredds catalog. A copy has been added to
            # the repository for just such an occasion. In the future, this
            # may be changed to be the default method
            expath = (
                'https://raw.githubusercontent.com/barronh/airfuse/main/grid/'
                + f'{gridkey}_GRID.nc'
            )
            r = requests.get(expath)
            r.raise_for_status()
            with open(gridpath, 'wb') as gridf:
                gridf.write(r.content)

    gridds = xr.open_dataset(gridpath).load()
    return gridds


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
    Finds and opens the most recent NCEI archived forecast.

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
    bdate = pd.to_datetime(bdate)
    edate = bdate + pd.to_timedelta('1h')
    if filedate is None:
        filedate = bdate
    if key.startswith('LOPZ99'):
        newkey = key.replace('LOPZ99', 'LZQZ99')
        logging.info(
            f'Archived BC-corrected ({key}) not available;'
            + f' will be replaced with {newkey}'
        )
        key = newkey
    paths = getpaths(filedate, service='dodsC', key=key)
    if len(paths) == 0:
        # allow failback, but warn when a day in the NCEI archive
        # is missing.
        logger.info(f'Could not find relevant file for {filedate:%FT%H}Z.')
    for path in paths[::-1]:
        try:
            naqfcf = xr.open_dataset(path)
            naqfcf = naqfcf.sel(time=edate, sigma=1)
            # Move "time" to midpoint, which helps prevent ambigous start/end
            naqfcf.coords['time'] = (
                naqfcf.coords['time'] + pd.to_timedelta('-30min')
            )
            addcrs(naqfcf)
            naqfcf.attrs['file_url'] = path
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
    bdate, key='LZQZ99_KWBP', filedate=None, source='nomads', failback='24h',
    verbose=4
):
    """
    Finds and opens the most recent NCEP (today or yesterday) or NWS (today
    only) forecast.

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
        Source either 'nws', 'ncep', 'noaa-nws-naqfc-pds.s3', or 'nomads' and
        only applies when requesting a file from the last two days.
        * 'nws' is the true operational site. (last two days only)
        * 'ncep' provides more thorough file naming. (last two days only)
        * 'nomads' is like 'ncep'
        * 'noaa-nws-naqfc-pds.s3' access via s3 bucket (2020-01-01-present)
          https://registry.opendata.aws/noaa-nws-naqfc-pds/

    Results
    -------
    outf : xarray.Dataset
        Outputs a file that looks like the NCEI archive file opened as a
        NetCDF file. In addition, it will have a crs_proj4 attrribute that
        describes the projection of the underlying file.

    Notes
    -----
    The noaa-nws-naqfc-pds.s3 is a new archive that holds the whole history
    """
    import os
    import pandas as pd
    import requests
    import tempfile
    import xarray as xr
    import pyproj
    import numpy as np

    if key.startswith('LZQZ99') or key.startswith('LOPZ99'):
        oldkey = 'pmtf'
        varkey = 'Particulate_matter_fine_sigma_1_Hour_Average'
        nwscode = 'apm25h01'
        ncepcode = 'ave_1hr_pm25'
        if key.startswith('LOPZ99'):
            nwscode = nwscode + '_bc'
            ncepcode = ncepcode + '_bc'
    elif key.startswith('LYUZ99') or key.startswith('YBPZ99'):
        oldkey = 'ozcon'
        varkey = 'Ozone_Concentration_sigma_1_Hour_Average'
        nwscode = 'ozone01'
        ncepcode = 'ave_1hr_o3'
        if key.startswith('YBPZ99'):
            nwscode = nwscode + '_bc'
            ncepcode = ncepcode + '_bc'
    else:
        raise KeyError(f'Unknown key {key}')

    bdate = pd.to_datetime(bdate)
    edate = bdate + pd.to_timedelta('1h')
    if filedate is None:
        filedate = bdate.floor('1d')

    gridds = getgrid(key)

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
    # 0 and 18Z were discontinued https://www.weather.gov/media/notification/
    #     pdf_2023_24/pns24-14_aqm_v7_product_removal.pdf
    for sh in [12, 6]:
        firsth = filedate + pd.to_timedelta(sh + 1, unit='h')
        lasth = filedate + pd.to_timedelta(
            {18: 6, 12: 72, 6: 72, 0: 6}[sh] + 1, unit='h'
        )
        if firsth > edate:
            continue
        if lasth < edate:
            continue
        # dt = bdate - filedate - pd.to_timedelta(sh, unit='h')
        # fh = round(dt.total_seconds() / 3600, 0)
        if source == 'ncep':
            url = (
                'https://ftp.ncep.noaa.gov/data/nccf/com/aqm/prod/'
                + f'aqm.{filedate:%Y%m%d}/{sh:02d}/'
                + f'aqm.t{sh:02d}z.{ncepcode}.227.grib2'
            )
        elif source == 'nomads':
            url = (
                'https://nomads.ncep.noaa.gov/pub/data/nccf/com/aqm/prod/'
                + f'aqm.{filedate:%Y%m%d}/{sh:02d}/'
                + f'aqm.t{sh:02d}z.{ncepcode}.227.grib2'
            )
        elif source == 'noaa-nws-naqfc-pds.s3':
            s3root = 'https://noaa-nws-naqfc-pds.s3.amazonaws.com'
            if filedate >= pd.to_datetime('2024-05-14T00Z'):
                s3root = f'{s3root}/AQMv7'
            elif filedate >= pd.to_datetime('2021-07-20T00Z'):
                s3root = f'{s3root}/AQMv6'
            elif filedate >= pd.to_datetime('2020-01-01T00Z'):
                s3root = f'{s3root}/AQMv5'
            else:
                raise KeyError(f'No {filedate}; AWS 2020-01-01 to present')
            url = (
                f'{s3root}/CS/{filedate:%Y%m%d}/{sh:02d}/'
                + f'aqm.t{sh:02d}z.{ncepcode}.{filedate:%Y%m%d}.227.grib2'
            )
        elif source == 'nws':
            url = (
                'https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/'
                + f'DC.ndgd/GT.aq/AR.conus/ds.{nwscode}.bin'
            )
        if verbose > 0:
            logger.info(url)

        tf = None
        try:
            if verbose > 1:
                logger.info(f'URL: {url}')
            r = requests.get(url)
            if r.status_code != 200:
                if verbose > 0:
                    logger.info(f'Code {r.status_code} {url}')
                continue
            # Windows requires delete=False to open the file a second time
            with tempfile.NamedTemporaryFile(delete=False) as tf:
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
                outf = f.drop_vars('valid_time').rename(**renames)
                if 'mask' in gridds.data_vars:
                    outf = outf.where(gridds['mask'] > 0)
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
                outf.attrs['file_url'] = url
                return outf
        except requests.models.HTTPError as e:
            print(str(e))
            continue
        except KeyError:
            # When 00 or 18Z are run, they only have 6 hours of data, which may
            # not include the file
            continue
        except Exception as e:
            raise e
        finally:
            # Ensure that the temporary file unlinked
            if tf:
                os.unlink(tf.name)
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
    If within the last two days, open operational forecast.
    Otherwise, open the most recent archive from NCEI's opendap.

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
        opts = dict(key=key, failback=failback, verbose=verbose)
        if ds < (1.5 * 24 * 3600):
            # put s3 last since it may incur NOAA a cost.
            servers = ['nomads', 'ncep', 'nws', 'noaa-nws-naqfc-pds.s3']
        else:
            # only s3 has a historical archive
            servers = ['noaa-nws-naqfc-pds.s3']
        # Cascaded fail system: first nomads, then ncep, then s3, then nws.
        # If any succeeds, the rest are not tried.
        for src in servers:
            if verbose > 0:
                logger.info(f'Calling open_operational {key} and {src}')
            try:
                naqfcf = open_operational(bdate, source=src, **opts)
                break
            except Exception as e:
                logger.info(f'{src} failed: {str(e)}')
        else:
            logger.info(f'open_operational with {servers} all failed.')
            logger.info(f'Calling open_mostrecent {key} (NCEI)')
            # consider adding a getgrid(key) and mask option for consistency
            # with open_operational
            naqfcf = open_mostrecent(
                bdate.replace(tzinfo=None), key=key, failback=failback
            )
        if path is not None:
            naqfcf.to_netcdf(path)

    if key.startswith('LZQZ99') or key.startswith('LOPZ99'):
        varkey = 'Particulate_matter_fine_sigma_1_Hour_Average'
    elif key.startswith('LYUZ99') or key.startswith('YBPZ99'):
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
    nowstr = pd.to_datetime('now', utc=True).strftime('%Y-%m-%dT%H:%M:%S')
    fileurl = naqfcf.attrs['file_url']
    var.attrs['description'] = f'{fileurl} (retrieved: {nowstr}Z)'
    return var
