__all__ = ['addattrs']
__doc__ = """Currently unused, but the goal is to easily convert dataframe
results from interpolation to gridded self-describing NetCDF files.
"""

_namer = {
    'NAQFC': 'NOAA Air Quality Forecast Capability',
    'VB': 'Voronoi Bias Corrected', 'VO': 'Voronoi Observation Interpolation',
    'VR': 'Voronoi Ratio Corrected', 'VQ': 'Voronoi NAQFC Interpolation',
    'aVNA': 'Voronoi Bias Corrected',
    'VNAO': 'Voronoi Observation Interpolation',
    'eVNA': 'Voronoi Ratio Corrected', 'VNAQ': 'Voronoi NAQFC Interpolation',
    'NB': 'Nearest Bias Corrected', 'NO': 'Nearest Observation Interpolation',
    'NR': 'Nearest Ratio Corrected', 'NQ': 'Nearest NAQFC Interpolation',
    'aIDW': 'Nearest Bias Corrected',
    'IDWO': 'Nearest Observation Interpolation',
    'eIDW': 'Nearest Ratio Corrected', 'IDWQ': 'Nearest NAQFC Interpolation',
    'AN': 'AirNow', 'PA': 'PurpleAir', 'NOAA': 'NOAA',
    'RF': 'Random Forest', 'FUSED': 'Fusion of enseble',
    'DIST': ' - Distance to nearest',
    'alpha0': 'First model fusion', 'alpha1': 'Second model fusion',
    'alpha2': 'Third model fusion', 'alpha3': 'Fourth model fusion',
    'WGT': 'Weighting scalar', 'BC': 'Bias Corrector',
    'SIMPLE': 'IDW and Logistic fusion', 'RF': 'Random Forest',
    'GW': 'Geographically Varying'
}


def addattrs(outds, units, spc, encoding=None):
    """
    Arguments
    ---------
    outds : xr.Dataset
    units : str
        Units for all variables
    spc : str
        Name of the species being processed (ozone or pm25)
    encoding : None or dict
        Dictionary of encoding properties to be used for the output

    Returns
    -------
    None
    """
    import warnings
    if encoding is None:
        encoding = dict(zlib=True, complevel=1)

    for key, dvar in outds.data_vars.items():
        if '_' in key:
            srckey, key = key.split('_')
        elif key == 'NAQFC':
            srckey = 'NOAA'
        elif key.startswith('alpha'):
            srckey = 'GVW'
        elif key in ('aVNA', 'eVNA', 'aIDW', 'eIDW'):
            srckey = ''
        else:
            warnings.warn(f'Unknown key: {key}')

        dvar.encoding.update(encoding)
        dvar.attrs.update(
            units=units, long_name=f'{key} {spc}',
            description=_namer.get(srckey, srckey) + ' ' + _namer.get(key, key)
        )
