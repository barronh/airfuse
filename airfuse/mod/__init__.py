__all__ = ['get_model']

from .naqfc import get_mostrecent as get_naqfc
from .geoscf import get_mostrecent as get_geoscf
from .goes import get_goesgwr


def get_model(date, key='o3', bbox=None, model='naqfc'):
    model = model.upper()
    if model == 'NAQFC':
        if key in ('o3', 'ozone'):
            key = 'LYUZ99_KWBP'
        elif key == 'pm25':
            key = 'LZQZ99_KWBP'
        var = get_naqfc(date, key=key)
    elif model == 'GEOSCF':
        if key in ('o3', 'ozone'):
            key = 'o3'
        elif key == 'pm25':
            key = 'pm25_rh35_gcc'
        var = get_geoscf(date, key='pm25_rh35_gcc', bbox=bbox)
    elif model == 'GOES':
        assert (key == 'pm25')
        var = get_goesgwr(date, key=key, bbox=bbox)
    else:
        raise KeyError(f'{model} unknown')

    var.name = model
    return var
