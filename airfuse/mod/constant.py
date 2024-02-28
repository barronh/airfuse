__all__ = ['get_constant']


def get_constant(
    bdate, key='o3', bbox=None, failback='24h', path=None, verbose=0,
    default=0.
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
    default : float
        Constant default value

    Returns
    -------
    var : xr.DataArray
        DataArray with values at cell centers and a projection stored as
        the attribute crs_proj4
    """
    from .naqfc import getgrid, addcrs
    import numpy as np
    import pandas as pd
    import xarray as xr
    import pyproj

    gridds = getgrid()
    addcrs(gridds)
    vals = np.zeros((gridds.sizes['y'], gridds.sizes['x']), dtype='f')
    vals[:] = default
    gridds['constant'] = (('y', 'x'), vals)
    var = gridds['constant']
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
    var.attrs['description'] = f'constant ({default}) on the NAQFC grid'
    return var
