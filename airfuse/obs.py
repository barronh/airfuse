__all__ = ['pair_airnow', 'pair_purpleair']


def pair_airnow(bdate, bbox, proj, var, spc):
    """
    pair_airnow calls pair_fem with src='airnow'

    See pair_fem for description of keywords.
    """
    return pair_fem(bdate, bbox, proj, var, spc, 'airnow')


def pair_aqs(bdate, bbox, proj, var, spc):
    """
    pair_aqs calls pair_fem with src='aqs'

    See pair_fem for description of keywords.
    """
    return pair_fem(bdate, bbox, proj, var, spc, 'aqs')


def pair_fem(bdate, bbox, proj, var, spc, src):
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
    return andf.query(f'{var.name} == {var.name}').copy()


def pair_purpleair(bdate, bbox, proj, var, spc, api_key=None):
    """
    Arguments
    ---------
    bdate : datelike
        Beginning date for PurpleAir observations. edate = bdate + 3599 seconds
    bbox : tuple
        lower left lon, lower left lat, upper right lon, upper right lat
    proj : pyproj.Proj
        Projection of the model variable (var)
    var : xr.DataArray
        Model variable with values on centers
    spc : str
        Name of the species to retrieve from AQS via pyrsig
    api_key : str or None
        If a str, it must either be the API key or a path to a file with only
        the API key in its contents.
        If None, the API key must exist in ~/.purpleairkey

    Returns
    -------
    padf : pandas.DataFrame
        Dataframe with values for at least (x, y, spc, Model, BIAS, RATIO)
        filtered so that only rows with Model are returned. This is important
        because otherwise eVNA and aVNA are invalid. The returned dataframe
        was preprocessed to average observations within half-sized grid cells
        before pairing with the model.
    """
    import pyrsig
    import pandas as pd
    import numpy as np
    import os

    assert(spc == 'pm25')
    outdir = f'{bdate:%Y/%m/%d}'
    bdate = pd.to_datetime(bdate)
    edate = bdate + pd.to_timedelta('3599s')
    if api_key is None:
        keypath = os.path.expanduser('~/.purpleairkey')
        if os.path.exists(keypath):
            api_key = open(keypath, 'r').read().strip()
        else:
            msgtxt = f'api_key must be provided or available at {keypath}'
            raise ValueError(msgtxt)
    elif os.path.exists(api_key):
        api_key = open(api_key, 'r').read().strip()

    rsigapi = pyrsig.RsigApi(
        bbox=bbox, workdir=outdir
    )
    rsigapi.purpleair_kw['api_key'] = api_key
    padf = rsigapi.to_dataframe(
        'purpleair.pm25_corrected', bdate=bdate, edate=edate,
        unit_keys=False, parse_dates=True
    ).rename(columns=dict(pm25_corrected_hourly=spc))

    # A much more complex analysis of error effectively only excludes
    # values over 1000. For simplicity, here we exclude measurements
    # greater than or equal too 1000 micrograms/m3.
    padf = padf.loc[~padf[spc].isnull()].query(f'{spc} < 1000').copy()

    padf['x'], padf['y'] = proj(
        padf['LONGITUDE'].values, padf['LATITUDE'].values
    )

    # Calculate PA values at 1/2 sized grid boxes (2.5km)
    hcell = 5.079 / 4
    xbins = np.linspace(
        var.x.min() - hcell, var.x.max() + hcell, var.x.size * 2
    )
    ybins = np.linspace(
        var.y.min() - hcell, var.y.max() + hcell, var.y.size * 2
    )

    col = pd.cut(padf['x'], xbins).apply(lambda x: x.mid).astype('f')
    col.name = 'COL'
    row = pd.cut(padf['y'], ybins).apply(lambda x: x.mid).astype('f')
    row.name = 'ROW'
    paadf = padf.groupby([col, row]).agg(
        x=('x', 'mean'), y=('y', 'mean'),
        COUNT=('COUNT', 'sum'), pm25=('pm25', 'mean')
    ).query(f'{spc} > 0').reset_index()
    paadf[var.name] = var.sel(
        x=paadf['x'].to_xarray(), y=paadf['y'].to_xarray(), method='nearest'
    ).values
    paadf['BIAS'] = paadf[var.name] - paadf[spc]
    paadf['RATIO'] = paadf[var.name] / paadf[spc]

    return paadf.query(f'{var.name} == {var.name}').copy()
