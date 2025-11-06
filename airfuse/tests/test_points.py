import os
import pytest
_bbox = (-80, 35, -55, 50)
_haspakey = os.path.exists(os.path.expanduser('~/.purpleairkey'))


def _test_airnowapi(**kwds):
    import tempfile
    from ..points import airnowapi
    with tempfile.TemporaryDirectory() as td:
        an = airnowapi(**kwds, inroot=td)
        df = an.get('2025-07-01')
        assert (df.shape[0] > 0)
        assert ('obs' in df.columns)
    return df


def _test_rsigairnow(**kwds):
    import tempfile
    from ..points import rsigairnow
    with tempfile.TemporaryDirectory() as td:
        an = rsigairnow(**kwds, inroot=td)
        df = an.get('2025-07-01')
        assert (df.shape[0] > 0)
        assert ('obs' in df.columns)
    return df


def _test_rsigpurpleair(**kwds):
    import tempfile
    from ..points import rsigpurpleair
    with tempfile.TemporaryDirectory() as td:
        an = rsigpurpleair(**kwds, inroot=td)
        df = an.get('2025-07-01')
        assert (df.shape[0] > 0)
        assert ('obs' in df.columns)
    return df


def test_get_airnowapi_pm25_hourly():
    _test_airnowapi(spc='pm25', nowcast=False, bbox=_bbox)


def test_get_airnowapi_pm25_nowcast():
    _test_airnowapi(spc='pm25', nowcast=True, bbox=_bbox)


def test_get_airnowapi_ozone_hourly():
    _test_airnowapi(spc='ozone', nowcast=False, bbox=_bbox)


def test_get_rsigairnow_pm25_hourly():
    _test_rsigairnow(spc='pm25', nowcast=False, bbox=_bbox)


def test_get_rsigairnow_pm25_nowcast():
    _test_rsigairnow(spc='pm25', nowcast=True, bbox=_bbox)


def test_get_rsigairnow_ozone_hourly():
    _test_rsigairnow(spc='ozone', nowcast=False, bbox=_bbox)


@pytest.mark.skipif(not _haspakey, reason="requires ~/.purpleairkey")
def test_get_rsigpurpleair_pm25_hourly():
    opts = dict(spc='pm25', nowcast=False, bbox=_bbox)
    igdf = _test_rsigpurpleair(dust='ignore', **opts)
    exdf = _test_rsigpurpleair(dust='exclude', **opts)
    assert igdf.shape[0] > exdf.shape[0]


def test_pair_rsigairnow():
    import tempfile
    from ..points import rsigairnow
    from ..layers import naqfc
    date = '2025-07-01'
    with tempfile.TemporaryDirectory() as td:
        opts = dict(spc='pm25', nowcast=False, inroot=td)
        an = rsigairnow(**opts)
        lay = naqfc(**opts)
        modvar = lay.get(date)
        padf = an.pair(date, modvar, proj=lay.proj)
        assert (padf.shape[0] > 0)
        assert ('obs' in padf.columns)
        assert ('mod' in padf.columns)
        assert ('x' in padf.columns)
        assert ('y' in padf.columns)


@pytest.mark.skipif(not _haspakey, reason="requires ~/.purpleairkey")
def test_pair_rsigpurpleair():
    import tempfile
    from ..points import rsigpurpleair
    from ..layers import naqfc
    date = '2025-07-01'
    with tempfile.TemporaryDirectory() as td:
        opts = dict(spc='pm25', nowcast=False, inroot=td)
        an = rsigpurpleair(**opts)
        lay = naqfc(**opts)
        modvar = lay.get(date)
        padf = an.pair(date, modvar, proj=lay.proj)
        assert (padf.shape[0] > 0)
        assert ('obs' in padf.columns)
        assert ('mod' in padf.columns)
        assert ('x' in padf.columns)
        assert ('y' in padf.columns)
