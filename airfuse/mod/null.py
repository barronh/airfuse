__all__ = ['get_zeros']


def get_zeros(
    bdate, key='o3', bbox=None, failback='24h', path=None, verbose=0
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
        Not used.
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
    from .naqfc import getgrid, addcrs
    import numpy as np
    import pandas as pd
    import xarray as xr
    import pyproj


    gridds = getgrid()
    addcrs(gridds)
    gridds['zero'] = (('y', 'x'), np.zeros((gridds.sizes['y'], gridds.sizes['x'])))
    var = gridds['zero']
    if bbox is not None:
        # Find lon/lat coordinates of projected cell centroids
        proj = pyproj.Proj(gridds.attrs['crs_proj4'])
        Y, X = xr.broadcast(var.y, var.x)
        LON, LAT = proj(X.values, Y.values, inverse=True)
        LON = X * 0 + LON
        LAT = Y * 0 + LAT
        # Find projected box covering lon/lat box
        inlon = ((LON >= bbox[0]) & (LON <= bbox[2])).any('y')
        inlat = ((LAT >= bbox[1]) & (LAT <= bbox[3])).any('x')
        var = var.sel(x=inlon, y=inlat)

    var.name = key
    var.coords['reftime'] = pd.to_datetime('now', utc=True)
    var.coords['sigma'] = 1.0
    var.coords['time'] = pd.to_datetime(bdate)
    var.attrs['crs_proj4'] = gridds.crs_proj4
    var.attrs['description'] = 'zeros on the NAQFC grid'
    return var
