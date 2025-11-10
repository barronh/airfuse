_varattrs = {
    "valid_time": {
        "var_desc": "valid_time",
        "units": "seconds since 1970-01-01T00:00:00+0000"
    },
    "time": {"units": "seconds since 1970-01-01T00:00:00+0000"},
    "longitude": {"units": "degrees_east"},
    "latitude": {"units": "degrees_north"},
    "naqfc": {
        "var_desc": "NOAA Air Quality Forecast Model with Bias Correction",
    },
    "obsx": {"units": "m"},
    "obsy": {"units": "m"},
    "x": {"units": "m"},
    "y": {"units": "m"},
    "obs": {"var_desc": "Observation value."},
    "mod": {"var_desc": "Model from original gridded product."},
    "mod_mbc": {"var_desc": "mod * obshat / modhat"},
    "mod_abc": {"var_desc": "mod + obshat - modhat"},
    "mod_bc": {
        "var_desc": (
            "abc = mod + obshat - modhat;"
            + " mbc = mod * obshat / modhat;"
            + " bc = where(abc<0,mbc,mean(abc,mbc))"
        )
    },
    "bc{yhatsfx}": {
        "var_desc": "mean(mod + obshat - modhat, mod * obshat / modhat)"
    },
    "abc{yhatsfx}": {"var_desc": "mod + obshat - modhat"},
    "mbc{yhatsfx}": {"var_desc": "mod * obshat / modhat"},
    "bc{cvsfx}": {
        "var_desc": "mean(mod + obshat - modhat, mod * obshat / modhat)"
    },
    "abc{cvsfx}": {"var_desc": "mod + obshat - modhat"},
    "mbc{cvsfx}": {"var_desc": "mod * obshat / modhat"},
    "mod_bc_cv": {"var_desc": "cross-validation of mod_bc"},
    "groups": {"var_desc": "observation group", "units": "none"},
    "sample_weight": {"var_desc": "sample weight", "units": "1"},
}
_fdesc = """Fusion of observations (e.g., AirNow and PurpleAir) using bias
correction of the NOAA NAQFC forecast model. The bias is
estimated in real-time using measurements. It is interpolated using the average
of nearest neighbors using extra weighting of Delaunay Neighbors.
"""


def addattrs(tgtds, units='micrograms/m**3', defattrs=None):
    import copy
    from .. import __version__
    import pandas as pd
    nowstr = pd.to_datetime('now', utc=True).strftime('%Y-%m-%dT%H:%M:%S%z')
    tgtds.attrs.update({
        'title': f'AirFuse ({__version__})',
        'author': 'Barron H. Henderson',
        'institution': 'US Environmental Protection Agency',
        'description': _fdesc, 'updated': nowstr
    })
    if defattrs is None:
        defattrs = copy.deepcopy(_varattrs)
    for k, v in tgtds.data_vars.items():
        # define default attributes
        dattrs = defattrs.get(
            k, {'long_name': k, 'units': units, 'var_desc': k}
        )
        # overwrite with any existing data
        dattrs.update(v.attrs)
        # reset attributes with defaults and original
        v.attrs.update(dattrs)


def df2ds(
    tgtdf, obdf, yhatsfx='_dnr', cvsfx='_dnr_cv', crs_proj4='',
    unit='micrograms/m**3'
):
    import copy
    tgtds = tgtdf.to_xarray()
    tgtds['obsx'] = obdf['x'].to_xarray()
    tgtds['obsy'] = obdf['y'].to_xarray()
    okeys = ['obs', 'mod_bc_cv', 'groups', 'sample_weight']
    for k in okeys:
        if k in obdf.columns:
            tgtds[k] = obdf[k].to_xarray()
    vo = dict(yhatsfx=yhatsfx, cvsfx=cvsfx)
    defattrs = copy.deepcopy(_varattrs)
    for vk, attrs in list(defattrs.items()):
        defattrs[vk.format(**vo)] = {
            k: v.format(**vo) for k, v in attrs.items()
        }

    addattrs(tgtds, crs_proj4=crs_proj4, units=unit, defattrs=defattrs)
