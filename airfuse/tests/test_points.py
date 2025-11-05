import os
import pytest
_bbox = (-80, 35, -55, 50)


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


def test_rsigairnow_pm25_hourly():
    _test_rsigairnow(spc='pm25', nowcast=False, bbox=_bbox)


_haspakey = os.path.exists(os.path.expanduser('~/.purpleairkey'))


@pytest.mark.skipif(not _haspakey, reason="requires ~/.purpleairkey")
def test_rsigpurpleair_pm25_hourly():
    opts = dict(spc='pm25', nowcast=False, bbox=_bbox)
    igdf = _test_rsigpurpleair(dust='ignore', **opts)
    exdf = _test_rsigpurpleair(dust='exclude', **opts)
    assert igdf.shape[0] > exdf.shape[0]


def test_rsigairnow_pm25_nowcast():
    _test_rsigairnow(spc='pm25', nowcast=True, bbox=_bbox)


def test_rsigairnow_ozone_hourly():
    _test_rsigairnow(spc='ozone', nowcast=False, bbox=_bbox)
