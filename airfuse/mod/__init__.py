__all__ = ['get_model']
__doc__ = """
Mod
---

This module provides functionality for retrieving NAQFC, GEOS-CF, and GOES
model data in a format that AirFuse can work with.
"""
from .naqfc import get_mostrecent as get_naqfc
from .geoscf import get_mostrecent as get_geoscf
from .goes import get_goesgwr
from .constant import get_constant


def get_model(date, key='o3', bbox=None, model='naqfc', verbose=0):
    """
    Get a model object for date and key that is optionally restricted to bbox
    Arguments
    ---------
    date : datetime-like
        Start time of the hour to retrieve from the model.
    key : str
        Species (o3 or pm25) to retrieve from the model.
    bbox : list
        swlon, swlat, nelon, nelat in decimal degrees (only supported by naqfc)
    model : str
        Which model to get: naqfc, geoscf, goes (case insensitive)
        - naqfc : NOAA Air Quality Forecast Capability
        - geoscf : NASA Goddard Earth Observing System Composition Forecast
        - goes : NOAA Geostationary Operational Environmental Satellites PM25
                 estimated from geographic weighted regression and deep neural
                 network correction.
    verbose : int
        Level of verbosity

    Returns
    -------
    var : xarray.DataArray
        Variable with crs_proj4 attribute defining the model projection.
    """
    model = model.upper()
    if model == 'NAQFC':
        if key in ('o3', 'ozone'):
            key = 'YBPZ99_KWBP'
        elif key == 'pm25':
            key = 'LOPZ99_KWBP'
        var = get_naqfc(date, key=key, bbox=bbox, verbose=verbose)
    elif model == 'GEOSCF':
        if key in ('o3', 'ozone'):
            key = 'o3'
        elif key == 'pm25':
            key = 'pm25_rh35_gcc'
        var = get_geoscf(date, key=key, bbox=bbox)
    elif model == 'GOES':
        assert (key == 'pm25')
        var = get_goesgwr(date, key=key, bbox=bbox, verbose=verbose)
    elif model == 'NULL':
        var = get_constant(
            date, key=key, bbox=bbox, verbose=verbose, default=0.
        )
    else:
        raise KeyError(f'{model} unknown')

    var.name = model
    return var
