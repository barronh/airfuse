import logging

logger = logging.getLogger(__name__)


def pair_goes(bdate, bbox, proj, var, spc, verbose=1, goeskey='bothdnn'):
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
    goeskey : str
        Must be either pm25gwr_ge or pm25gwr_gw or both or
        pm25dnn_ge or pm25dnn_gw or bothdnn
    Returns
    -------
    andf : pandas.DataFrame
        Dataframe with values for at least (x, y, spc, Model, BIAS, RATIO)
        filtered so that only rows with Model are returned. This is important
        because otherwise eVNA and aVNA are invalid. The returned dataframe
        was preprocessed to average observations within half-sized grid cells
        before pairing with the model.
    """
    import pyproj
    import pandas as pd
    from ..mod import get_goesgwr

    assert (spc == 'pm25')

    # For a single variable, set appropriate dimensions
    # For both or bothdnn, apply to individual variables and concatenate.
    if goeskey in ('pm25gwr_ge', 'pm25dnn_ge'):
        xkey = 'xdim_ge'
        ykey = 'ydim_ge'
    elif goeskey in ('pm25gwr_gw', 'pm25dnn_gw'):
        xkey = 'xdim_gw'
        ykey = 'ydim_gw'
    elif goeskey == 'both':
        gedf = pair_goes(bdate, bbox, proj, var, spc, verbose, 'pm25gwr_ge')
        gwdf = pair_goes(bdate, bbox, proj, var, spc, verbose, 'pm25gwr_gw')
        return pd.concat([gedf, gwdf], ignore_index=True)
    elif goeskey == 'bothdnn':
        gedf = pair_goes(bdate, bbox, proj, var, spc, verbose, 'pm25dnn_ge')
        gwdf = pair_goes(bdate, bbox, proj, var, spc, verbose, 'pm25dnn_gw')
        return pd.concat([gedf, gwdf], ignore_index=True)
    else:
        raise KeyError('must be either pm25gwr_ge or pm25gwr_gw')

    goesv = get_goesgwr(bdate, key=spc, varkey=goeskey, bbox=bbox).rename(
        x=xkey, y=ykey
    )
    gproj = pyproj.Proj(goesv.attrs['crs_proj4'])

    goesvdf = goesv.to_dataframe().dropna()
    gx = goesvdf.index.get_level_values(xkey)
    gy = goesvdf.index.get_level_values(ykey)

    glon, glat = gproj(gx.values, gy.values, inverse=True)

    goesvdf['x'], goesvdf['y'] = proj(glon, glat)
    xidx = goesvdf.reset_index()['x'].to_xarray()
    yidx = goesvdf.reset_index()['y'].to_xarray()
    goesvdf[var.name] = var.sel(x=xidx, y=yidx, method='nearest').values
    goesvdf[spc] = goesvdf[goeskey]

    # replace with concatenation of east and west.
    df = goesvdf
    df['BIAS'] = df[var.name] - df[spc]
    df['RATIO'] = df[var.name] / df[spc]
    return df.query(f'{var.name} == {var.name}').reset_index()
