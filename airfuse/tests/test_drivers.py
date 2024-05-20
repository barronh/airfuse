import warnings
import pandas as pd
from ..drivers import fuse

recentdate = (
    pd.to_datetime('now', utc=True).floor('1h') - pd.to_timedelta('3h')
)

olddate = (
    pd.to_datetime('now', utc=True).floor('1d') - pd.to_timedelta('15d')
    + pd.to_timedelta('18h')
)


def test_ozone():
    outpaths = fuse(
        'airnow', 'o3', olddate, 'naqfc',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    warnings.warn(str(outpaths))


def test_pmairnow():
    outpaths = fuse(
        'airnow', 'pm25', olddate, 'naqfc',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    warnings.warn(str(outpaths))


def test_pmpurpleair():
    outpaths = fuse(
        'airnow', 'pm25', olddate, 'naqfc',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    warnings.warn(str(outpaths))


def test_ozonenrt():
    outpaths = fuse(
        'airnow', 'o3', recentdate, 'naqfc',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    warnings.warn(str(outpaths))


def test_ozonegeoscf():
    warnings.warn('Bypassing GEOS-CF test due to instability')
    return
    outpaths = fuse(
        'airnow', 'o3', olddate, 'geoscf',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    warnings.warn(str(outpaths))
