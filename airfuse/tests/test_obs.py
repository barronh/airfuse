def get_dummyvar():
    """
    Produce a dummy variable that mimics NAQFC
    """
    import numpy as np
    import xarray as xr

    srs = (
        '+proj=lcc +lat_1=25 +lat_0=25 +lon_0=265 +k_0=1 +x_0=0 +y_0=0'
        + ' +R=6371229 +to_meter=1000 +no_defs'
    )
    vals = np.ones((1025, 1473), dtype='f')
    dy = dx = 5.079
    x = np.arange(-4226.1084, 3250.1794 + dx, dx)
    y = np.arange(-832.6978, 4368.198 + dy, dy)
    var = xr.DataArray(vals, dims=('y', 'x'), coords=dict(x=x, y=y))
    var.attrs['crs_proj4'] = srs
    var.name = 'NAQFC'
    return var


def _test_srcspc(spc, src):
    import pyproj
    import numpy as np
    from ..obs.epa import pair_rsig, pair_airnowrsig, pair_aqsrsig

    var = get_dummyvar()
    proj = pyproj.Proj(var.attrs['crs_proj4'])
    bdate = '2021-08-14T18Z'
    bbox = (-97, 25, -67, 50)
    df1 = pair_rsig(bdate, bbox, proj, var, spc, src)
    if src == 'airnow':
        df2 = pair_airnowrsig(bdate, bbox, proj, var, spc)
    elif src == 'aqs':
        df2 = pair_aqsrsig(bdate, bbox, proj, var, spc)

    if src in ('airnow', 'aqs'):
        assert (df1 == df2).replace(np.nan, -999).all().all()


def test_pm():
    _test_srcspc('pm25', 'airnow')
    _test_srcspc('pm25', 'aqs')


def test_ozone():
    _test_srcspc('ozone', 'airnow')
    _test_srcspc('ozone', 'aqs')


def test_airnowapi():
    from ..obs import epa
    import pyproj
    import pandas as pd

    date = pd.to_datetime('2024-02-28T18Z')
    obskey = 'pm25'
    bbox = (-130, 20, -60, 55)

    modvar = get_dummyvar()
    proj = pyproj.Proj(modvar.crs_proj4)
    obsdf0 = epa.pair_airnowapi(date, bbox, proj, modvar, obskey, montype=0)
    obsdf2 = epa.pair_airnowapi(date, bbox, proj, modvar, obskey)
    assert (obsdf2.shape[0] >= obsdf0.shape[0])


def test_airnowaqobsfile():
    from ..obs import epa
    import pyproj
    import pandas as pd

    date = pd.to_datetime('2024-01-01T00Z')
    obskey = 'pm25'
    bbox = (-130, 20, -60, 55)

    modvar = get_dummyvar()
    proj = pyproj.Proj(modvar.crs_proj4)
    obsdf0 = epa.pair_airnowaqobsfile(date, bbox, proj, modvar, obskey)
    assert (obsdf0.shape[0] > 0)


def test_airnowhourlydatafile():
    from ..obs import epa
    import pyproj
    import pandas as pd

    date = pd.to_datetime('2024-01-01T00Z')
    obskey = 'pm25'
    bbox = (-130, 20, -60, 55)

    modvar = get_dummyvar()
    proj = pyproj.Proj(modvar.crs_proj4)
    obsdf0 = epa.pair_airnowhourlydatafile(date, bbox, proj, modvar, obskey)
    assert (obsdf0.shape[0] > 0)
