import pytest
import pandas as pd

recentdate = (
    pd.to_datetime('now', utc=True).floor('1h') - pd.to_timedelta('3h')
)

olddate = (
    pd.to_datetime('now', utc=True).floor('1d') - pd.to_timedelta('15d')
    + pd.to_timedelta('18h')
)


def test_get_constant():
    from ..mod.constant import get_constant
    from ..mod import get_model
    date = '2024-06-01'
    ones = get_model(date, key='o3', model='NULL', verbose=0) + 1
    twos = get_constant(date, bbox=(-85, 35, 65, 50), default=2)
    assert (twos / ones).mean() == 2
    assert ones.size > twos.size


@pytest.mark.xfail(strict=False)
def test_naqfc():
    """
    NAQFC should give the same values from all data sources.
    nomads, ncep, and nws should be sufficiently updated to match.
    """
    from ..mod.naqfc import open_operational
    import numpy as np

    noaakey = 'LZQZ99_KWBP'
    varkey = 'Particulate_matter_fine_sigma_1_Hour_Average'
    f1 = open_operational(recentdate, key=noaakey, source='nomads')
    f2 = open_operational(recentdate, key=noaakey, source='ncep')
    f3 = open_operational(recentdate, key=noaakey, source='nws')
    v1 = np.ma.masked_invalid(f1[varkey])
    v2 = np.ma.masked_invalid(f2[varkey])
    v3 = np.ma.masked_invalid(f3[varkey])
    assert v1.shape == v2.shape
    assert np.ma.allclose(v1, v2)
    if v1.shape == v3.shape:
        assert np.ma.allclose(v1, v3)


@pytest.mark.xfail(strict=False)
def test_ncei():
    """
    NCEI can be flaky because the OpenDAP server gets clogged up.
    Results should match other sources, but not testing.
    """
    from ..mod.naqfc import open_mostrecent

    noaakey = 'LZQZ99_KWBP'
    varkey = 'Particulate_matter_fine_sigma_1_Hour_Average'
    f1 = open_mostrecent(olddate.replace(tzinfo=None), key=noaakey)
    assert f1.sizes['x'] == 1473
    assert f1.sizes['y'] == 1025
    assert varkey in f1


@pytest.mark.xfail(strict=False)
def test_goes():
    from ..mod.goes import get_goesgwr
    # yesterday at noon central
    goesdate = recentdate.floor('1d') - pd.to_timedelta('6h')
    fge = get_goesgwr(goesdate, key='pm25', varkey='pm25dnn_ge')
    fgw = get_goesgwr(goesdate, key='pm25', varkey='pm25dnn_gw')
    assert fge.sizes['y'] == 1500
    assert fge.sizes['x'] == 2500
    assert fgw.sizes['y'] == 1200
    assert fgw.sizes['x'] == 1800


@pytest.mark.xfail(strict=False)
def test_geoscf():
    from ..mod import get_model
    f = get_model(
        recentdate, key='o3', bbox=(-90, 30, -80, 40), model='geoscf'
    )
    assert f.sizes == {'y': 161, 'x': 161}


@pytest.mark.xfail
def test_get_bad():
    from ..mod import get_model
    get_model('2024-06-01', key='o3', model='oops', verbose=0)


@pytest.mark.xfail
def test_get_goesfail():
    from ..mod import get_model
    get_model('2024-06-01', key='o3', model='GOES')


def test_getgrid():
    from ..mod.naqfc import getgrid
    f = getgrid()
    assert f.sizes['y'] == 1025
    assert f.sizes['x'] == 1473
