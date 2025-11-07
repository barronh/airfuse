from scipy.special import erfc
import xarray as xr
import numpy as np
import pandas as pd


# %
# Synthetic World Approximating Sample Error
# ------------------------------------------
#

np.random.seed(44)
ds = xr.Dataset()
ds.coords['x'] = np.linspace(0, 1, 101)
ds.coords['y'] = np.linspace(0, 1, 101)
ds['truth'] = (
    ('y', 'x'),
    30 * (1 + np.sin(ds.x * 3 * np.pi) * np.sin(ds.y * 3 * np.pi)).data
)
# Random Error is proportional to 5th percentile value
err = float(ds.truth.quantile(0.05))
ds['z'] = np.maximum(
    0, ds['truth'] + np.random.normal(loc=0, scale=err, size=ds['truth'].shape)
)
dx = float(ds.x.diff('x').mean())
dy = float(ds.y.diff('y').mean())
hyp = (dx**2 + dy**2)**.5

xkeys = ['x', 'y']
ykeys = ['z', 'mod']
# Convert ds to dataframe
df = ds.stack(p=xkeys).to_dataframe().reset_index(drop=True)

# %
# Sample Networks
# ---------------
# - Increase sampling frequency based on value (more dense near urban)
# - Allow different frequency by network

# identify
aqspct = 2000 / 459 / 356
papct = 10000 / 459 / 356
rank = df.z.rank()


def sample(df, scale, pct, m, b, err):
    chk = np.random.randint(0, df.shape[0], size=df.shape[0]) * scale
    fdf = df.loc[df.z.rank() > chk]
    subidx = np.arange(fdf.shape[0])
    np.random.shuffle(subidx)
    subidx = subidx[:int(pct * df.shape[0])]
    fdf = fdf.iloc[subidx].copy()
    fdf['z'] = fdf['z'] * m + b + np.random.normal(
        loc=0, scale=err / 2, size=fdf.shape[0]
    )
    return fdf


# AQS/AirNow have zero systematic absolute bias
# AQS/AirNow have zero systematic relative bias
andf = sample(df, 2, aqspct, 1, 0, err / 2)
xm = andf.z.mean()
xs = andf.z.std()
modf = erfc(-(df.z - xm) / xs) / 2.5
df['mod'] = xm * modf + xm - xs * 1.5
andf = df.loc[andf.index].copy()

# PurpleAir have are low-biased by 1 unit.
# PurpleAir have a 5% systematic relative low-bias
padf = sample(df, 1.5, papct, 0.95, -1, err)


obdf = pd.concat([andf, padf], keys=['an', 'pa'], ignore_index=False)
obdf.index.names = ['groups', 'index']
obdf['groups'] = obdf.index.to_frame()['groups']
obdf['sample_weight'] = obdf['groups'].map(
    lambda x: {'pa': 0.25, 'an': 1}[x]
)


def test_dnr():
    from ..utils import fuse, mpestats
    from .. import dnr
    # %
    # Create Target and CV Predictions
    # --------------------------------
    tgtdf = df
    # ds.sel(
    #     x=slice(None, None, 10), y=slice(None, None, 10)
    # ).stack(p=('y', 'x')).to_dataframe().reset_index(drop=True)

    sdnr = dnr.DelaunayNeighborsRegressor(
        n_neighbors=30, delaunay_weights='only',
        weights=lambda x: np.maximum(x, .0001)**-2
    )
    gdnr = dnr.GroupedDelaunayNeighborsRegressor(
        n_neighbors=30, delaunay_weights='only', weights={
            'an': lambda d: np.maximum(.0001, d)**-2,
            'pa': lambda d: np.maximum(.4, d)**-2,
        }
    )

    def anfw(dnr, X):
        mind = dnr.kneighbors(X, n_neighbors=1)[0].min(1)
        return np.maximum(mind, dx / 4)**-2

    def pafw(dnr, X):
        mind = dnr.kneighbors(X, n_neighbors=1)[0].min(1)
        return 0.25 * np.maximum(mind, dx / 2)**-2

    fdnr = dnr.FusedDelaunayNeighborsRegressor(
        n_neighbors=30, delaunay_weights='only',
        weights=lambda x: np.maximum(x, .0001)**-2,
        fusion_weight={'an': anfw, 'pa': pafw}
    )
    fuse(
        tgtdf, andf, obdnr=sdnr, yhatsfx='_an_dnr', cvsfx='_an_dnr_cv',
        obskey='z'
    )
    for k in ['bc', 'mbc', 'abc']:
        idx = obdf.groups == 'an'
        col = f'{k}_an_dnr_cv'
        obdf.loc[idx, col] = andf[f'{k}_an_dnr_cv'].values

    fuse(
        tgtdf, padf, obdnr=sdnr, yhatsfx='_pa_dnr', cvsfx='_pa_dnr_cv',
        obskey='z'
    )
    fuse(obdf, None, obdnr=sdnr, yhatsfx='_pa_dnr', cvsfx=None, obskey='z')
    fitkwds = dict(sample_weight=obdf['sample_weight'], groups=obdf['groups'])
    fuse(
        tgtdf, obdf, obdnr=gdnr, yhatsfx='_ob_gdnr', cvsfx='_ob_gdnr_cv',
        obskey='z'
    )
    fuse(
        tgtdf, obdf, obdnr=fdnr, yhatsfx='_ob_fdnr', cvsfx='_ob_fdnr_cv',
        obskey='z', fitkwds=fitkwds
    )
    dropkeys = ['x', 'y', 'groups', 'sample_weight']
    statdf = mpestats(
        obdf.query('groups == "an"').drop(dropkeys, axis=1), refkey='z'
    ).T
    # last known valid test
    refds = pd.Series({
        'bc_an_dnr_cv': 3.53285559, 'mbc_an_dnr_cv': 3.55427420,
        'abc_an_dnr_cv': 3.54590187, 'mbc_ob_gdnr_cv': 3.61433850,
        'abc_ob_gdnr_cv': 3.59600817, 'bc_ob_gdnr_cv': 3.58633836,
        'mbc_ob_fdnr_cv': 3.85927064, 'abc_ob_fdnr_cv': 3.85857435,
        'bc_ob_fdnr_cv': 3.85257722
    }, name='rmse')
    chkds = statdf['rmse'][refds.index].round(8)
    pd.testing.assert_series_equal(refds, chkds)
