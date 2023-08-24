import warnings
import pandas as pd
from ..drivers import fuse


def test_ozone():
    outpaths = fuse(
        'airnow', 'o3', pd.to_datetime('2023-08-24T18Z'), 'naqfc',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    warnings.warn(str(outpaths))


def test_pmairnow():
    outpaths = fuse(
        'airnow', 'pm25', pd.to_datetime('2023-08-24T18Z'), 'naqfc',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    warnings.warn(str(outpaths))


def test_pmpurpleair():
    outpaths = fuse(
        'airnow', 'pm25', pd.to_datetime('2023-08-24T18Z'), 'naqfc',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    warnings.warn(str(outpaths))


def test_ozonegeoscf():
    outpaths = fuse(
        'airnow', 'o3', pd.to_datetime('2023-08-24T18Z'), 'geoscf',
        (-97, 25, -67, 50), cv_only=True, overwrite=True
    )
    warnings.warn(str(outpaths))
