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
    from ..obs import pair_fem, pair_airnow, pair_aqs

    var = get_dummyvar()
    proj = pyproj.Proj(var.attrs['crs_proj4'])
    bdate = '2021-08-14T18Z'
    bbox = (-97, 25, -67, 50)
    df1 = pair_fem(bdate, bbox, proj, var, spc, src)
    if src == 'airnow':
        df2 = pair_airnow(bdate, bbox, proj, var, spc)
    elif src == 'aqs':
        df2 = pair_aqs(bdate, bbox, proj, var, spc)

    if src in ('airnow', 'aqs'):
        assert (df1 == df2).replace(np.nan, -999).all().all()


def test_pm():
    _test_srcspc('pm25', 'airnow')
    _test_srcspc('pm25', 'aqs')


def test_ozone():
    _test_srcspc('ozone', 'airnow')
    _test_srcspc('ozone', 'aqs')
