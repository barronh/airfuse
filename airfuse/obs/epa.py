__all__ = [
    'pair_airnow', 'pair_aqs',
    'pair_airnowapi', 'pair_aqsapi',
    'pair_airnowrsig', 'pair_aqsrsig',
    'pair_airnowaqobsfile', 'pair_airnowhourlydatafile'
]

import logging
logger = logging.getLogger(__name__)


def pair_airnow(bdate, bbox, proj, var, spc, api_key=None, verbose=1):
    """
    Currently uses pair_airnowapi if within 2 days and pair_airnowaqobsfile
    otherwise.

    Arguments
    ---------
    bdate : datelike
        Beginning date for AirNow observations. edate = bdate + 3599 seconds
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    proj : pyproj.Proj
        Projection of the model variable (var)
    var : xr.DataArray
        Model variable with values on centers
    spc : str
        Name of the species to retrieve from AirNow via AirNowAPI
    api_key : str
        API key or path to file with api_key for airnow.
    verbose : int
        Level of verbosity

    Returns
    -------
    andf : pandas.DataFrame
        Dataframe with values for at least (x, y, spc, Model, BIAS, RATIO)
        filtered so that only rows with Model are returned. This is important
        because otherwise eVNA and aVNA are invalid. The returned dataframe
        was preprocessed to average observations within half-sized grid cells
        before pairing with the model.
    """
    import pandas as pd

    bdate = pd.to_datetime(bdate)
    dt = bdate - pd.to_datetime('now', utc=True)
    twodaysago = (-2 * 24 * 3600)
    pkwds = dict(
        bdate=bdate, bbox=bbox, proj=proj, var=var, spc=spc
    )
    if dt.total_seconds() < twodaysago:
        if verbose > 0:
            logger.info('pair_airnow using AirNow File')
        odf = pair_airnowaqobsfile(**pkwds)
    else:
        if verbose > 0:
            logger.info('pair_airnow using AirNow API')
        try:
            odf = pair_airnowapi(api_key=api_key, verbose=verbose, **pkwds)
        except KeyError:
            logger.warning(
                'pair_airnow using RSIG API because api_key is None and'
                ' ~/.airnowkey does not exist'
            )
            odf = pair_airnowrsig(**pkwds)

    return odf


def pair_aqs(
    bdate, bbox, proj, var, spc, verbose=1, api_user=None, api_key=None
):
    """
    Currently calls pair_aqsrsig
    Arguments
    ---------
    bdate : datelike
        Beginning date for AirNowAPI observations. edate = bdate + 3599 seconds
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    proj : pyproj.Proj
        Projection of the model variable (var)
    var : xr.DataArray
        Model variable with values on centers
    spc : str
        Name of the species to retrieve from AQS via AQS API
    api_user: str
        Username for the AQS API
    api_key : str or None
        If a str, it must either be the API key or a path to a file with only
        the API key in its contents.
        If None, the API key must exist in ~/.aqskey
    verbose : int
        Level of verbosity

    Returns
    -------
    aqsdf : pandas.DataFrame
        Dataframe with values for at least (x, y, spc, Model, BIAS, RATIO)
        filtered so that only rows with Model are returned. This is important
        because otherwise eVNA and aVNA are invalid. The returned dataframe
        was preprocessed to average observations within half-sized grid cells
        before pairing with the model.
    """
    pkwds = dict(
        bdate=bdate, bbox=bbox, proj=proj, var=var, spc=spc
    )
    try:
        odf = pair_aqsapi(api_user=api_user, api_key=api_key, **pkwds)
    except KeyError:
        logger.warn(
            'pair_aqs using RSIG API because api_user or api_key are None'
            ' and ~/.aqskey does not exist'
        )
        odf = pair_aqsrsig(**pkwds)
    return odf


def pair_airnowapi(
    bdate, bbox, proj, var, spc, api_key=None, montype=0, verbose=1
):
    """
    Get obs from AirNowAPI and pair with model variable (var)

    Arguments
    ---------
    bdate : datelike
        Beginning date for AirNowAPI observations. edate = bdate + 3599 seconds
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    proj : pyproj.Proj
        Projection of the model variable (var)
    var : xr.DataArray
        Model variable with values on centers
    spc : str
        Name of the species to retrieve from AirNow via AirNowAPI
    api_key : str or None
        If a str, it must either be the API key or a path to a file with only
        the API key in its contents.
        If None, the API key must exist in ~/.airnowkey
    montype : int
        0 - regulatory monitors (default)
        1 - mobile monitors
        2 - both
    verbose : int
        Level of verbosity

    Returns
    -------
    andf : pandas.DataFrame
        Dataframe with values for at least (x, y, spc, Model, BIAS, RATIO)
        filtered so that only rows with Model are returned. This is important
        because otherwise eVNA and aVNA are invalid. The returned dataframe
        was preprocessed to average observations within half-sized grid cells
        before pairing with the model.
    """
    import requests
    import pandas as pd
    import os
    import numpy as np

    if api_key is None:
        # default to home ~/.airnowkey
        keypath = os.path.expanduser('~/.airnowkey')
        if os.path.exists(keypath):
            api_key = keypath
        else:
            msgtxt = f'api_key must be provided or available at {keypath}'
            raise KeyError(msgtxt)
    if os.path.exists(api_key):
        api_key = open(api_key).read().strip()

    bdate = pd.to_datetime(bdate)
    edate = bdate + pd.to_timedelta('3599s')
    if montype in (1, 2):
        now = pd.to_datetime('now', utc=True).floor('1d')
        dt = (now - bdate)
        dtd = dt.total_seconds() / 3600 / 24
        if verbose > 0 and dtd > 2:
            logger.warning(
                'pair_airnowapi using mobile monitors more than 2 days old;'
                + ' historic locations of mobile monitors via api are not'
                + ' reliable.'
            )
    bbox_str = '{},{},{},{}'.format(*bbox)
    r = requests.get(
        'https://www.airnowapi.org/aq/data/?'
        f'startDate={bdate:%Y-%m-%dT%H}&endDate={edate:%Y-%m-%dT%H}'
        + f'&parameters={spc.upper()}&BBOX={bbox_str}&'
        + f'dataType=C&format=application/json&verbose=1&monitorType={montype}'
        + f'&includerawconcentrations=1&API_KEY={api_key}')
    df = pd.DataFrame.from_records(r.json())
    df = df.replace(-999., np.nan)
    df['x'], df['y'] = proj(df['Longitude'].values, df['Latitude'].values)
    df[var.name] = var.sel(
        x=df['x'].to_xarray(), y=df['y'].to_xarray(), method='nearest'
    ).values
    df[spc] = df['RawConcentration']
    df['BIAS'] = df[var.name] - df[spc]
    df['RATIO'] = df[var.name] / df[spc]
    return df.query(f'{var.name} == {var.name} and {spc} == {spc}')


def pair_airnowaqobsfile(bdate, bbox, proj, var, spc):
    """
    Get obs from AirNow AWS HourlyAQObs*.dat and pair with model variable (var)

    In some cases, HourlyAQObs may have data that is  missing in HourlyData.
    HourlyData is updated for 48h after the hour twice each hour at 25 and 55
    min past the hour. HourlyAQObs is updated for 72h after the hour at 35 min
    past the hour.

    Arguments
    ---------
    bdate : datelike
        Beginning date for AirNowAPI observations. edate = bdate + 3599 seconds
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    proj : pyproj.Proj
        Projection of the model variable (var)
    var : xr.DataArray
        Model variable with values on centers
    spc : str
        Name of the species to retrieve from AirNow via AirNow Hourly AQObs

    Returns
    -------
    andf : pandas.DataFrame
        Dataframe with values for at least (x, y, spc, Model, BIAS, RATIO)
        filtered so that only rows with Model are returned. This is important
        because otherwise eVNA and aVNA are invalid. The returned dataframe
        was preprocessed to average observations within half-sized grid cells
        before pairing with the model.
    """
    import pandas as pd

    url = (
        'https://files.airnowtech.org/airnow/'
        + f'{bdate:%Y/%Y%m%d/HourlyAQObs_%Y%m%d%H}.dat'
    )
    spckey = {'pm25': 'PM25', 'ozone': 'OZONE'}[spc.lower()]
    df = pd.read_csv(url, encoding='latin1').query(
        f'{spckey}_Measured == 1 and {spckey} == {spckey}'
        + f' and Latitude >= {bbox[1]} and Latitude <= {bbox[3]}'
        + f' and Longitude >= {bbox[0]} and Longitude <= {bbox[2]}'
    )
    df['x'], df['y'] = proj(df['Longitude'].values, df['Latitude'].values)
    df[var.name] = var.sel(
        x=df['x'].to_xarray(), y=df['y'].to_xarray(), method='nearest'
    ).values
    df[spc] = df[spckey]
    df['BIAS'] = df[var.name] - df[spc]
    df['RATIO'] = df[var.name] / df[spc]

    return df.query(f'{var.name} == {var.name} and {spc} == {spc}')


def pair_airnowhourlydatafile(bdate, bbox, proj, var, spc):
    """
    Get obs from AirNow HourlyData*.dat and pair with model variable (var)

    In some cases, HourlyAQObs may have data that is  missing in HourlyData.
    HourlyData is updated for 48h after the hour twice each hour at 25 and 55
    min past the hour. HourlyAQObs is updated for 72h after the hour at 35 min
    past the hour.

    Arguments
    ---------
    bdate : datelike
        Beginning date for AirNowAPI observations. edate = bdate + 3599 seconds
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    proj : pyproj.Proj
        Projection of the model variable (var)
    var : xr.DataArray
        Model variable with values on centers
    spc : str
        Name of the species to retrieve from AirNow via AirNow HourlyData file

    Returns
    -------
    andf : pandas.DataFrame
        Dataframe with values for at least (x, y, spc, Model, BIAS, RATIO)
        filtered so that only rows with Model are returned. This is important
        because otherwise eVNA and aVNA are invalid. The returned dataframe
        was preprocessed to average observations within half-sized grid cells
        before pairing with the model.
    """
    import pandas as pd
    spckey = {'pm25': 'PM2.5'}[spc]
    airnowroot = 'https://files.airnowtech.org/airnow'
    airnowsitecols = (
        'AQSID|parameter_name|site_code|site_name|status|agency_id'
        + '|agency_name|EPA_region|Latitude|Longitude|elevation|GMT_offset'
        + '|country_code|d1|d2|MSA_code|MSA_name|state_code|state_name'
        + '|county_code|county_name|d3|d4'
    ).split('|')
    sitemetadf = pd.read_csv(
        f'{airnowroot}/{bdate:%Y/%Y%m%d}/monitoring_site_locations.dat',
        encoding='latin1', delimiter='|',
        names=airnowsitecols
    ).query('parameter_name == "PM2.5"').groupby('AQSID').first().reset_index()

    airnowobscols = (
        "valid_date|valid_time|AQSID|sitename|GMT_offset|parameter_name"
        + "|reporting_units|obs_value|data_source"
    ).split('|')
    obsonlydf = pd.read_csv(
        f'{airnowroot}/{bdate:%Y/%Y%m%d/HourlyData_%Y%m%d%H}.dat',
        encoding='latin1', delimiter='|', names=airnowobscols
    ).query(
        f'parameter_name == "{spckey}"'
    )
    obsonlydf['AQSID'] = obsonlydf['AQSID'].astype(str)
    sitemetadf['AQSID'] = sitemetadf['AQSID'].astype(str)
    df = obsonlydf.merge(
        sitemetadf.loc[:, ['AQSID', 'agency_name', 'Latitude', 'Longitude']],
        on='AQSID', how='inner'
    ).query(
        f'parameter_name == "{spckey}"'
        + f' and Latitude >= {bbox[1]} and Latitude <= {bbox[3]}'
        + f' and Longitude >= {bbox[0]} and Longitude <= {bbox[2]}'
    )
    df['x'], df['y'] = proj(df['Longitude'].values, df['Latitude'].values)
    df[var.name] = var.sel(
        x=df['x'].to_xarray(), y=df['y'].to_xarray(), method='nearest'
    ).values
    df[spc] = df['obs_value']
    df['BIAS'] = df[var.name] - df[spc]
    df['RATIO'] = df[var.name] / df[spc]

    return df.query(f'{var.name} == {var.name} and {spc} == {spc}')


def pair_airnowrsig(bdate, bbox, proj, var, spc):
    """
    pair_airnow calls pair_rsig with src='airnow'

    See pair_rsig for description of keywords.
    """
    return pair_rsig(bdate, bbox, proj, var, spc, 'airnow')


def pair_aqsrsig(bdate, bbox, proj, var, spc):
    """
    pair_aqs calls pair_rsig with src='aqs'

    See pair_rsig for description of keywords.
    """
    return pair_rsig(bdate, bbox, proj, var, spc, 'aqs')


def pair_aqsapi(bdate, bbox, proj, var, spc, api_user=None, api_key=None):
    """
    Arguments
    ---------
    bdate : datelike
        Beginning date for AirNowAPI observations. edate = bdate + 3599 seconds
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    proj : pyproj.Proj
        Projection of the model variable (var)
    var : xr.DataArray
        Model variable with values on centers
    spc : str
        Name of the species to retrieve from AQS via AQS API
    api_user: str
        Username for the AQS API
    api_key : str or None
        If a str, it must either be the API key or a path to a file with only
        the API key in its contents.
        If None, the API key must exist in ~/.aqskey

    Returns
    -------
    aqsdf : pandas.DataFrame
        Dataframe with values for at least (x, y, spc, Model, BIAS, RATIO)
        filtered so that only rows with Model are returned. This is important
        because otherwise eVNA and aVNA are invalid. The returned dataframe
        was preprocessed to average observations within half-sized grid cells
        before pairing with the model.
    """
    import requests
    import pandas as pd
    import os
    import numpy as np
    from ..util import read_netrc

    if api_key is None:
        keypath = os.path.expanduser('~/.aqskey')
        netrcpath = os.path.expanduser('~/.netrc')
        netrcwinpath = os.path.expanduser('~/_netrc')
        if os.path.exists(keypath):
            api_key = keypath
        elif os.path.exists(netrcpath):
            api_key = netrcpath
        elif os.path.exists(netrcwinpath):
            api_key = netrcwinpath
        else:
            raise KeyError(
                'api_key must either be provided as a .netrc style file at'
                + f' {api_key} or ~/.netrc or ~/_netrc with user and password'
                + ' for aqs.epa.gov'
            )

    api_user, dummy, api_key = read_netrc(api_key, 'aqs.epa.gov')

    params = {'ozone': [44201], 'pm25': []}[spc]
    dfs = []
    for param in params:
        r = requests.get(
            'https://aqs.epa.gov/data/api/sampleData/byBox'
            + f'?email={api_user}&key={api_key}&'
            + f'&param={param}&bdate={bdate:%Y%m%d}&edate={bdate:%Y%m%d}'
            + '&minlat={1}&maxlat={3}&minlon={0}&maxlon={2}'.format(*bbox)
        )
        r.raise_for_status()
        jdata = r.json()
        defhead = [{'status': 'failed', 'error': 'no header'}]
        header = jdata.get('Header', defhead)[0]
        if header.get('status', 'failed').lower() == 'failed':
            raise IOError(header['error'])
        df = pd.DataFrame.from_records(jdata['Data'])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.replace(-999., np.nan)
    df['x'], df['y'] = proj(df['longitude'].values, df['latitude'].values)
    df[var.name] = var.sel(
        x=df['x'].to_xarray(), y=df['y'].to_xarray(), method='nearest'
    ).values
    df[spc] = df['sample_measurement']
    if spc == 'ozone':
        factor = df['units_of_measure'].apply(
            lambda x: {'Parts per million': 1000}.get(x, 1)
        )
        df[spc] = df[spc] * factor

    df['BIAS'] = df[var.name] - df[spc]
    df['RATIO'] = df[var.name] / df[spc]

    return df.query(f'{var.name} == {var.name} and {spc} == {spc}')


def pair_rsig(bdate, bbox, proj, var, spc, src):
    """
    Arguments
    ---------
    bdate : datelike
        Beginning date for AQS observations. edate = bdate + 3599 seconds.
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    proj : pyproj.Proj
        Projection of the model variable (var)
    var : xr.DataArray
        Model variable with values on centers
    spc : str
        Name of the species to retrieve from AQS via pyrsig

    Returns
    -------
    andf : pandas.DataFrame
        Dataframe with values for at least (x, y, spc, Model, BIAS, RATIO)
        filtered so that only rows with Model are returned. This is important
        because otherwise eVNA and aVNA are invalid.
    """
    import pyrsig
    import pandas as pd

    bdate = pd.to_datetime(bdate)
    edate = bdate + pd.to_timedelta('3599s')

    outdir = f'{bdate:%Y/%m/%d}'
    rsigapi = pyrsig.RsigApi(
        bbox=bbox, workdir=outdir
    )
    andf = rsigapi.to_dataframe(
        f'{src}.{spc}', bdate=bdate, edate=edate, unit_keys=False,
        parse_dates=True
    )
    andf = andf.loc[~andf[spc].isnull()].copy()

    andf['x'], andf['y'] = proj(
        andf['LONGITUDE'].values, andf['LATITUDE'].values
    )
    andf[var.name] = var.sel(
        x=andf['x'].to_xarray(), y=andf['y'].to_xarray(), method='nearest'
    ).values
    andf['BIAS'] = andf[var.name] - andf[spc]
    andf['RATIO'] = andf[var.name] / andf[spc]
    return andf.query(f'{var.name} == {var.name} and {spc} == {spc}').copy()
