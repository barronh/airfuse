import numpy as np
import pyproj
import logging

logger = logging.getLogger(__name__)


# Copying parameters from file in instead of reading each time
# Assumes that none of the coordinates change.
ge_proj_kw = dict(
    long_name="GOES-R ABI fixed grid projection",
    grid_mapping_name="geostationary",
    perspective_point_height=35786023.,
    semi_major_axis=6378137., semi_minor_axis=6356752.31414,
    inverse_flattening=298.2572221, latitude_of_projection_origin=0.,
    longitude_of_projection_origin=-75., sweep_angle_axis="x"
)
ge_proj = pyproj.Proj(pyproj.CRS.from_cf(ge_proj_kw))
ge_proj4 = ge_proj.srs

# Coordinates are shorts starting at 0 and ending at n - 1
# with CF variables defined below
# short y(y) ; (0 ... 1499)
# y:scale_factor = -5.6e-05f ;
# y:add_offset = 0.128212f ;
# y:units = "rad" ;
# short x(x) ; (0 ... 2499)
# x:scale_factor = 5.6e-05f ;
# x:add_offset = -0.101332f ;
# x:units = "rad" ;

i2rad = np.float32(5.6e-05)
sat_h = float(ge_proj_kw['perspective_point_height'])
gexoff = np.float32(-0.101332)
ge_x = (np.arange(2500, dtype=np.short) * i2rad + gexoff) * sat_h
geyoff = np.float32(0.128212)
ge_y = (np.arange(1500, dtype=np.short) * (-i2rad) + geyoff) * sat_h

# Copying parameters from file in instead of reading each time
# Assumes that none of the coordinates change.
gw_proj_kw = dict(
    long_name="GOES-R ABI fixed grid projection",
    grid_mapping_name="geostationary",
    perspective_point_height=35786023.,
    semi_major_axis=6378137., semi_minor_axis=6356752.31414,
    inverse_flattening=298.2572221, latitude_of_projection_origin=0.,
    longitude_of_projection_origin=-137., sweep_angle_axis="x"
)
gw_proj = pyproj.Proj(pyproj.CRS.from_cf(gw_proj_kw))
gw_proj4 = gw_proj.srs

# Coordinates are shorts starting at 0 and ending at n - 1
# with CF variables defined below
# short y(y) ; # (0 ... 5423)
# y:scale_factor = -5.6e-05f ;
# y:add_offset = 0.151844f ;
# y:units = "rad" ;
# short x(x) ; # (0 ... 5423)
# x:scale_factor = 5.6e-05f ;
# x:add_offset = -0.151844f ;
fdiscoff = np.float32(0.151844)
sat_h = float(gw_proj_kw['perspective_point_height'])
gwf_x = (np.arange(0, 5424, dtype=np.short) * i2rad - fdiscoff) * sat_h
gwf_y = (np.arange(0, 5424, dtype=np.short) * (-i2rad) + fdiscoff) * sat_h

# Hai Zhang per email 2023-08-14 at 8:45am Eastern
# pm25sat_gw is from AODF.  The indices ranges are [300:1500,2900:4700]
# for efficiency, we could change arange from 0, 5424 to these ranges
gw_x = gwf_x[2900:4700]
gw_y = gwf_y[300:1500]


def get_goesgwr(
    bdate, key='pm25', varkey='pm25gwr_ge', bbox=None, path=None
):
    from .util import get_file
    import pandas as pd

    bdate = pd.to_datetime(bdate)
    server = 'www.star.nesdis.noaa.gov'
    urlroot = f'https://{server}/pub/smcd/hzhang/GOES/GOES-16/NRT/CONUS'
    filename = f'{bdate:pm25gwr/%Y%m%d/pm25_gwr_aod_exp50_%Y%m%d%H.nc}'
    localpath = f'{bdate:%Y/%m/%d/pm25_gwr_aod_exp50_%Y%m%d%H.nc}'
    # filename = f'{bdate:pm25dnn/%Y%m%d/pm25_gwr_aod_exp50_%Y%m%d%H_dnn.nc}'
    # localpath = f'{bdate:%Y/%m/%d/pm25_gwr_aod_exp50_%Y%m%d%H_dnn.nc}'

    remotepath = f'{urlroot}/{filename}'
    get_file(remotepath, localpath)
    gwrf = open_goes(localpath)
    if varkey.endswith('_ge'):
        da = gwrf[varkey].rename(xdim_ge='x', ydim_ge='y')
        da.attrs['crs_proj4'] = ge_proj4
    elif varkey.endswith('_gw'):
        da = gwrf[varkey].rename(xdim_gw='x', ydim_gw='y')
        da.attrs['crs_proj4'] = gw_proj4
    else:
        raise KeyError(f'Variable keys must end in _ge or _gw; got {varkey}')

    return da


def open_goes(path):
    """
    Open a GEOS-PM25 GWR or DNN file and return it in the GWR file style

    Arguments
    ---------
    path : str
        Path to a pm25gwr or pm25dnn file.
        * GWR file must have pm25sat_ge and pm25sat_gw
        * DNN file must have pm25sat_com and pm25gwr_dnn_com

    Returns
    -------
    f : xarray.Dataset
        Output file with at least pm25gwr_ge and pm25gwr_gw variable
        and, if source is pm25dnn, it will also have pm25dnn_ge and
        pm25dnn_gw. All variables are aligned with the global variables
        ge_x, ge_y, gw_x and ge_y
    """
    import xarray as xr
    srcf = xr.open_dataset(path)
    if 'pm25sat_gw' in srcf and 'pm25sat_ge' in srcf:
        outf = srcf.rename(pm25sat_gw='pm25gwr_gw', pm25sat_ge='pm25gwr_ge')
    else:
        # Iteratively determined the starting/ending of arrays
        # Then, buffering the dimensions so that it is the same lengths
        edim = ('ydim_ge', 'xdim_ge')
        wdim = ('ydim_gw', 'xdim_gw')

        # Convenience function to shorten next few lines
        da = xr.DataArray
        se = (slice(122, None), slice(1254, None))
        sw = (slice(None, 1200), slice(None, 1254))
        datavars = dict(
            pm25gwr_ge=da(srcf['pm25sat_com'][se].values, dims=edim),
            pm25dnn_ge=da(srcf['pm25gwr_dnn_com'][se].values, dims=edim),
            pm25gwr_gw=da(srcf['pm25sat_com'][sw].values, dims=wdim),
            pm25dnn_gw=da(srcf['pm25gwr_dnn_com'][sw].values, dims=wdim),
        )
        coords = dict(
            xdim_ge=np.arange(2133), xdim_gw=np.arange(0, 1254),
        )

        outf = xr.Dataset(datavars, coords=coords).sel(
            xdim_ge=da(np.arange(-367, ge_x.size - 367), dims=('xdim_ge',)),
            xdim_gw=da(np.arange(0, gw_x.size), dims=('xdim_gw',)),
            method='nearest'
        )
    outf.coords['xdim_ge'] = ge_x
    outf.coords['ydim_ge'] = ge_y
    outf.coords['xdim_gw'] = gw_x
    outf.coords['ydim_gw'] = gw_y

    return outf
