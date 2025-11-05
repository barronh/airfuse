def _check(spc, date, nowcast, maxval):
    from ..layers import naqfc
    import numpy as np
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        lay = naqfc('pm25', nowcast=nowcast, maxval=maxval, inroot=td)
        v = lay.get(date)
        minv = v.min()
        maxv = v.min()
        assert (minv >= 0)
        assert (maxv <= maxval)
        np.testing.assert_equal([1025, 1473], [v.sizes[k] for k in ['y', 'x']])


def test_naqfc_ozone_hourly_now():
    import pandas as pd
    date = pd.to_datetime('now').floor('1h') - pd.to_timedelta('24h')
    _check('ozone', date, False, 2e3)


def test_naqfc_pm25_hourly_now():
    import pandas as pd
    date = pd.to_datetime('now').floor('1h') - pd.to_timedelta('24h')
    _check('pm25', date, False, 2e3)


def test_naqfc_pm25_nowcast_now():
    import pandas as pd
    date = pd.to_datetime('now').floor('1h') - pd.to_timedelta('24h')
    _check('pm25', date, True, 2e3)


def test_naqfc_ozone_hourly_past():
    date = '2020-01-01T18'
    _check('ozone', date, False, 2e3)


def test_naqfc_pm25_hourly_past():
    date = '2022-07-01T18'
    _check('pm25', date, False, 2e3)


def test_naqfc_pm25_nowcast_past():
    date = '2024-06-01T18'
    _check('pm25', date, True, 2e3)
