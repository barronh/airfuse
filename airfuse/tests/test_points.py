import os
import pytest
_bbox = (-80, 35, -55, 50)
_haspakey = os.path.exists(os.path.expanduser('~/.purpleairkey'))
_hasfasm = os.path.exists(os.path.expanduser('~/fasm.json'))


def _test_airnowapi(**kwds):
    import tempfile
    from ..points import airnowapi
    with tempfile.TemporaryDirectory() as td:
        an = airnowapi(**kwds, inroot=td)
        df = an.get('2025-07-01')
        assert (df.shape[0] > 0)
        assert ('obs' in df.columns)
    return df


def _test_airnowrsig(**kwds):
    import tempfile
    from ..points import airnowrsig
    with tempfile.TemporaryDirectory() as td:
        an = airnowrsig(**kwds, inroot=td)
        df = an.get('2025-07-01')
        assert (df.shape[0] > 0)
        assert ('obs' in df.columns)
    return df


def _test_purpleairrsig(**kwds):
    import tempfile
    from ..points import purpleairrsig
    with tempfile.TemporaryDirectory() as td:
        an = purpleairrsig(**kwds, inroot=td)
        df = an.get('2025-07-01')
        assert (df.shape[0] > 0)
        assert ('obs' in df.columns)
    return df


def _test_airnowfasm(**kwds):
    from ..points import airnowfasm
    import pandas as pd
    lag = pd.to_timedelta('1h')
    now = (pd.to_datetime('now', utc=True) - lag).floor('1h')
    an = airnowfasm(**kwds)
    df = an.get(now)
    assert (df.shape[0] > 0)
    assert ('obs' in df.columns)
    return df


def _test_purpleairfasm(**kwds):
    from ..points import purpleairfasm
    import pandas as pd
    lag = pd.to_timedelta('1h')
    now = (pd.to_datetime('now', utc=True) - lag).floor('1h')
    an = purpleairfasm(**kwds)
    df = an.get(now)
    assert (df.shape[0] > 0)
    assert ('obs' in df.columns)
    return df


def test_get_airnowapi_pm25_hourly():
    _test_airnowapi(spc='pm25', nowcast=False, bbox=_bbox)


def test_get_airnowapi_pm25_nowcast():
    _test_airnowapi(spc='pm25', nowcast=True, bbox=_bbox)


def test_get_airnowapi_ozone_hourly():
    _test_airnowapi(spc='ozone', nowcast=False, bbox=_bbox)


def test_get_airnowrsig_pm25_hourly():
    _test_airnowrsig(spc='pm25', nowcast=False, bbox=_bbox)


def test_get_airnowrsig_pm25_nowcast():
    _test_airnowrsig(spc='pm25', nowcast=True, bbox=_bbox)


def test_get_airnowrsig_ozone_hourly():
    _test_airnowrsig(spc='ozone', nowcast=False, bbox=_bbox)


@pytest.mark.skipif(not _haspakey, reason="requires ~/.purpleairkey")
def test_get_purpleairrsig_pm25_hourly():
    opts = dict(spc='pm25', nowcast=False, bbox=_bbox)
    igdf = _test_purpleairrsig(dust='ignore', **opts)
    exdf = _test_purpleairrsig(dust='exclude', **opts)
    assert igdf.shape[0] > exdf.shape[0]


def test_pair_airnowrsig():
    import tempfile
    from ..points import airnowrsig
    from ..layers import naqfc
    date = '2025-07-01'
    with tempfile.TemporaryDirectory() as td:
        opts = dict(spc='pm25', nowcast=False, inroot=td)
        an = airnowrsig(**opts)
        lay = naqfc(**opts)
        modvar = lay.get(date)
        padf = an.pair(date, modvar, proj=lay.proj)
        assert (padf.shape[0] > 0)
        assert ('obs' in padf.columns)
        assert ('mod' in padf.columns)
        assert ('x' in padf.columns)
        assert ('y' in padf.columns)


@pytest.mark.skipif(not _haspakey, reason="requires ~/.purpleairkey")
def test_pair_purpleairrsig():
    import tempfile
    from ..points import purpleairrsig
    from ..layers import naqfc
    date = '2025-07-01'
    with tempfile.TemporaryDirectory() as td:
        opts = dict(spc='pm25', nowcast=False, inroot=td)
        an = purpleairrsig(**opts)
        lay = naqfc(**opts)
        modvar = lay.get(date)
        padf = an.pair(date, modvar, proj=lay.proj)
        assert (padf.shape[0] > 0)
        assert ('obs' in padf.columns)
        assert ('mod' in padf.columns)
        assert ('x' in padf.columns)
        assert ('y' in padf.columns)


@pytest.mark.skipif(not _hasfasm, reason="requires ~/fasm.json")
def test_get_airnowfasm_pm25_hourly():
    _test_airnowfasm(spc='pm25', nowcast=False, bbox=_bbox)


@pytest.mark.skipif(not _hasfasm, reason="requires ~/fasm.json")
def test_get_airnowfasm_pm25_nowcast():
    _test_airnowfasm(spc='pm25', nowcast=True, bbox=_bbox)


@pytest.mark.skipif(not _hasfasm, reason="requires ~/fasm.json")
def test_get_purpleairfasm_pm25_hourly():
    _test_purpleairfasm(spc='pm25', nowcast=False, bbox=_bbox)


@pytest.mark.skipif(not _hasfasm, reason="requires ~/fasm.json")
def test_get_purpleairfasm_pm25_nowcast():
    _test_purpleairfasm(spc='pm25', nowcast=True, bbox=_bbox)
