__all__ = ['pair', 'pair_airnow', 'pair_aqs', 'pair_purpleair', 'pair_goes']
__doc__ = """
pair uses the src argument to indirectly call pair_<src> functions. Each
pair_<src> function returns a dataset paired with a model variable

s
"""

from .epa import pair_airnow, pair_aqs
from .purpleair import pair_purpleair
from .goes import pair_goes
import logging
logger = logging.getLogger(__name__)


def pair(bdate, bbox, proj, var, spc, src, **kwds):
    """
    Thin wrapper around pair_airnow, pair_aqs, pair_purpleair, and pair_goes
    pair and all pair_<src> functions all require a minimum set of arguments.
    Additional arguments are function specific and are passed to the src
    function by keyword arguments.

    Arguments
    ---------
    bdate : date-like
        Beginning hour of observational dataset
    bbox : tuple
        bounding box (wlon, slat, elon, nlat)
    proj : str or pyproj.Proj
        Projection of model variable.
    var : xr.DataArray
        Model variable with values on centers
    spc : str
        Species of observational dataset ('o3', 'pm25', etc)
    src : str
        choices 'airnow', 'aqs', 'purpleair', 'goes'
    kwds : mappable
        Passed to pair_<src> function

    Returns
    -------
    obsdf : pandas.DataFrame
        Dataframe with values for at least (x, y, spc, Model, BIAS, RATIO)
        where spc is the observed value, Model is from var, BIAS is Model
        minus spc, and RATIO is Model divided by spc. Rows are filtered so
        that only rows with Model are returned. This is important because
        otherwise eVNA and aVNA are invalid.

    Notes
    -----
    Currently implemented as individual functions, but could be updated as an
    object with acquire, pair and filter methods. Ideally, with a get method
    that wraps them all.
    """
    pair_func = {
        'airnow': pair_airnow,
        'aqs': pair_aqs,
        'purpleair': pair_purpleair,
        'goes': pair_goes,
    }.get(src, None)
    if src is None:
        msg = f'{src} unknown obs source; try airnow, aqs, purpleair, or goes'
        logger.error(msg)
        raise ValueError(msg)
    obsdf = pair_func(bdate, bbox, proj, var, spc, **kwds)
    return obsdf
