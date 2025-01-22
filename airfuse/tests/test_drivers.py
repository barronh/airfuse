import pytest
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
    # warnings.warn(str(outpaths))
    assert len(outpaths) > 0


def test_pmairnow():
    outpaths = fuse(
        'airnow', 'pm25', olddate, 'naqfc',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    # warnings.warn(str(outpaths))
    assert len(outpaths) > 0


@pytest.mark.xfail(strict=False, reason='PurpleAir requires API key')
def test_pmpurpleair():
    outpaths = fuse(
        'purpleair', 'pm25', olddate, 'naqfc',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    # warnings.warn(str(outpaths))
    assert len(outpaths) > 0


def test_ozonenrt():
    outpaths = fuse(
        'airnow', 'o3', recentdate, 'naqfc',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    # warnings.warn(str(outpaths))
    assert len(outpaths) > 0


def test_ozonetiny():
    outpaths = fuse(
        'airnow', 'o3', recentdate, 'naqfc',
        (-80, 38, -72, 41), cv_only=False, overwrite=True, format='nc'
    )
    # warnings.warn(str(outpaths))
    assert len(outpaths) > 0


@pytest.mark.xfail(strict=False, reason='GEOS-CF OpenDAP unreliable')
def test_ozonegeoscf():
    outpaths = fuse(
        'airnow', 'o3', olddate, 'geoscf',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    warnings.warn(str(outpaths))
