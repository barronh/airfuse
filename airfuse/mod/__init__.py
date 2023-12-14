__all__ = ['get_model']

from .naqfc import get_mostrecent as get_naqfc
from .geoscf import get_mostrecent as get_geoscf
from .goes import get_goesgwr


def get_model(date, key='o3', bbox=None, model='naqfc', verbose=0):
    """
    Arguments
    ---------
    date : datetime-like
        Start time of the hour to retrieve from the model.
    key : str
        Species (o3 or pm25) to retrieve from the model.
    bbox : list
        swlon, swlat, nelon, nelat in decimal degrees
    model : str
        Which mdoel to get: naqfc, geoscf, goes (case insensitive)
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
    else:
        raise KeyError(f'{model} unknown')

    var.name = model
    return var
