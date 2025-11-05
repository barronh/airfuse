_varattrs = {
    "valid_time": {
        "var_desc": "valid_time",
        "units": "seconds since 1970-01-01T00:00:00+0000"
    },
    "time": {
        "units": "seconds since 1970-01-01T00:00:00+0000"
    },
    "longitude": {
        "units": "degrees_east",
    },
    "latitude": {
        "units": "degrees_north",
    },
    "naqfc": {
        "var_desc": "NOAA Air Quality Forecast Model with Bias Correction",
    },
    "obsx": {
        "units": "m"
    },
    "obsy": {
        "units": "m"
    },
    "x": {
        "units": "m"
    },
    "y": {
        "units": "m"
    },
    "obs": {
        "var_desc": "Observation value."
    },
    "mod": {
        "var_desc": "Model from original gridded product."
    },
    "mod{yhatsfx}": {
        "var_desc": "Model interpolated from observational locations"
    },
    "obs{yhatsfx}": {
        "var_desc": "Interpolated Observations"
    },
    "mbc{yhatsfx}": {
        "var_desc": "mbc{yhatsfx} = mod * obs{yhatsfx} / mod{yhatsfx}"
    },
    "abc{yhatsfx}": {
        "var_desc": "abc{yhatsfx} = mod + obs{yhatsfx} - mod{yhatsfx}"
    },
    "bc{yhatsfx}": {
        "var_desc": "bc{yhatsfx} = mbc if abc < 0 else avg(mbc, abc)"
    },
    "mod{cvsfx}": {
        "var_desc": "Obs interpolated from training observations"
    },
    "obs{cvsfx}": {
        "var_desc": "Obs interpolated from training observations"
    },
    "mbc{cvsfx}": {
        "var_desc": "mbc{cvsfx} = mod * obs{cvsfx} / mod{cvsfx}"
    },
    "abc{cvsfx}": {
        "var_desc": "abc{cvsfx} = mod + obs{cvsfx} - mod{cvsfx}"
    },
    "bc{cvsfx}": {
        "var_desc": "bc{cvsfx} = mbc{cvsfx} if abc{cvsfx} < 0 else abc{cvsfx}"
    },
    "groups": {
        "var_desc": "observation group", "units": "none"
    },
    "weight": {
        "var_desc": "sample weight", "units": "1"
    },
}


def df2ds(
    gdf, obdf, yhatsfx, cvsfx, crs_proj4, unit='micrograms/m**3', varattrs=None
):
    from .. import __version__
    import pandas as pd
    nowstr = pd.to_datetime('now', utc=True).strftime('%Y-%m-%dT%H:%M:%S%z')
    dnrdesc = {
        '_dnr': 'DelaunayNeighborsRegressor',
        '_gdnr': 'GroupedDelaunayNeighborsRegressor',
        '_fdnr': 'FusedDelaunayNeighborsRegressor',
    }.get(yhatsfx, yhatsfx)
    fdesc = f"""Fusion of observations (AirNow and PurpleAir) using residual
interpolation and correction of the NOAA NAQFC forecast model. The bias is
estimated in real-time using AirNow and PurpleAir measurements. It is
interpolated using the average of nearest neighbors using weighting of
Delaunay Neighbors as implemented in {dnrdesc}.

Variable Name Explanations:
- obs: Observations
- mod: model
- obs{yhatsfx}: model interpolated from monitor locations.
- mod{yhatsfx}: model interpolated from monitor locations.
- abc{yhatsfx}: add bias correction (mod + obs{yhatsfx} - mod{yhatsfx})
- mbc{yhatsfx}: mult bias correction (mod * obs{yhatsfx} / mod{yhatsfx})

updated: {nowstr}
"""
    fileattrs = {
        'title': f'AirFuse ({__version__})',
        'author': 'Barron H. Henderson',
        'institution': 'US Environmental Protection Agency',
        'description': fdesc, 'crs_proj4': crs_proj4,
        'updated': nowstr
    }

    if varattrs is None:
        vo = dict(yhatsfx=yhatsfx, cvsfx=cvsfx)
        varattrs = {
            vk.format(**vo): {
                pk: (pv.format(**vo) if isinstance(pv, str) else pv)
                for pk, pv in vattrs.items()
            }
            for vk, vattrs in _varattrs.items()
        }
    for k, attrs in varattrs.items():
        attrs.setdefault('units', unit)
        attrs.setdefault('long_name', k)
    outds = gdf.to_xarray().transpose('time', 'y', 'x')
    gkeep = [k for k in list(outds.data_vars)]
    for gk in gkeep:
        if gk not in outds.dims and gk not in ('longitude', 'latitude'):
            outds[gk] = outds[gk].astype('f')
        outds[gk].attrs.update(varattrs.get(gk, {}))
    obdf = obdf.rename(columns=dict(x='obsx', y='obsy'))
    pkeep = [
        'obsx', 'obsy', 'obs', 'groups', 'weight', f'obs{cvsfx}',
        f'mod{cvsfx}', f'abc{cvsfx}', f'mbc{cvsfx}', f'bc{cvsfx}',
    ]
    pkeep = [k for k in pkeep if k in obdf.columns]
    obds = obdf[pkeep].to_xarray()
    for pk in pkeep:
        outds[pk] = obds[pk]
        outds[pk].attrs.update(varattrs.get(pk, {}))
    outds.attrs.update(fileattrs)

    return outds[gkeep + pkeep]
