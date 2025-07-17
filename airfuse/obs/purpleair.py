__all__ = ['pair_purpleair']


def pair_purpleair(
    bdate, bbox, proj, var, spc, api_key=None, exclude_stations=None,
    dust_ev_filt=False
):
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
    exclude_stations : None or list
        List of stations to exclude.
    dust_ev_filt : bool
        If True, this removes sites where a dust event is likely

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

    assert (spc == 'pm25')
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

    if dust_ev_filt is True:
        # Bring in PM 0.3um and PM 5um count data for dust event filtering
        pm03df = rsigapi.to_dataframe(
            'purpleair.0_3_um_count', bdate=bdate, edate=edate,
            unit_keys=False, parse_dates=True
        )

        pm5df = rsigapi.to_dataframe(
            'purpleair.5_um_count', bdate=bdate, edate=edate,
            unit_keys=False, parse_dates=True
        )

        # Merge PM 0.3um and PM5um counts and delete rows
        # with incomplete data between the two fields
        pm03df['new_index'] = (
            pm03df['STATION'].astype(str) + '_'
            + pm03df['Timestamp'].astype(str)
        )
        pm5df['new_index'] = (
            pm5df['STATION'].astype(str) + '_' + pm5df['Timestamp'].astype(str)
        )
        pm03df.set_index('new_index', inplace=True)
        pm5df.set_index('new_index', inplace=True)

        pm03df = pm03df[['0_3_um_count_hourly']]
        pm5df = pm5df[['5_um_count_hourly']]
        padf_dust = pd.merge(pm03df, pm5df, how='inner', on='new_index')

        # Calculate PM 0.3um counts/PM 5um counts (dust criteria) ratio
        padf_dust = padf_dust.astype({'0_3_um_count_hourly': float,
                                      '5_um_count_hourly': float})
        padf_dust['0.3um ct/5um ct'] = (
            padf_dust['0_3_um_count_hourly'] / padf_dust['5_um_count_hourly']
        )
        padf_dust.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Aggregate up to the hour for dust criteria ratio
        padf_dust.loc[:, 'STATION-TIME'] = padf_dust.index
        padf_dust['STATION-TIME'] = padf_dust['STATION-TIME'].str[:-11]
        padf_dust = padf_dust[['STATION-TIME', '0.3um ct/5um ct']]

        padf_dust = padf_dust.groupby('STATION-TIME',
                                      as_index=False).mean()
        padf_dust[['STATION', 'Timestamp(UTC) Hourly']] = (
            padf_dust['STATION-TIME'].str.split('_', expand=True)
        )
        padf_dust['STATION'] = padf_dust['STATION'].astype(np.int64)

        # Filter hourly pm25 for non-dust event measurements
        padf_dust = padf_dust[padf_dust['0.3um ct/5um ct'] > 190]

        padf_aft = pd.merge(padf, padf_dust, how='inner', on='STATION')
        removed_sta = len(padf) - len(padf_aft)
        print('%s monitors removed due to dust event' % removed_sta)
        padf_aft.drop(['Timestamp(UTC) Hourly', '0.3um ct/5um ct'],
                      axis=1, inplace=True)
        padf = padf_aft

    # Identify PurpleAir records to exclude
    # missing value or unreasonably high value
    exclude = padf[spc].isnull() | (padf[spc] >= 1000)
    # Exclude specific stations
    if exclude_stations is not None:
        # Ensure type consistency
        exstrs = [str(i) for i in exclude_stations]
        exclude = exclude | padf.STATION.astype(str).isin(exstrs)

    # A much more complex analysis of error effectively only excludes
    # values over 1000. For simplicity, here we exclude measurements
    # greater than or equal too 1000 micrograms/m3.
    padf = padf.loc[~exclude].copy()

    padf['x'], padf['y'] = proj(
        padf['LONGITUDE'].values, padf['LATITUDE'].values
    )
    # add a PurpleAir removal process

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
