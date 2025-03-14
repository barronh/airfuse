__all__ = [
    'get_file', 'wget_file', 'request_file', 'ftp_file', 'read_netrc',
    'mpestats', 'to_geopandas', 'to_geojson', 'df2nc', 'to_webpng'
]


def get_file(url, local_path, wget=False):
    """
    Download file from ftp or http via wget, ftp_file, or request_file

    Arguments
    ---------
    url : str
        Path on server
    local_path : str
        Path to save file (usually url without file protocol prefix
    wget : bool
        If True, use wget (default: False)

    Returns
    -------
    local_path : str
        local_path
    """
    if wget:
        return wget_file(url, local_path)
    elif url.startswith('ftp://'):
        return ftp_file(url, local_path)
    else:
        return request_file(url, local_path)


def ftp_file(url, local_path):
    """
    While files are on STAR ftp, use this function.

    Arguments
    ---------
    url : str
        Path on ftp server
    local_path : str
        Path to save file (usually url without ftp://)

    Returns
    -------
    local_path : str
        local_path
    """
    import ftplib
    import os

    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    server = url.split('//')[1].split('/')[0]
    remotepath = url.split(server)[1]
    ftp = ftplib.FTP(server)
    ftp.login()
    with open(local_path, 'wb') as fp:
        ftp.retrbinary(f'RETR {remotepath}', fp.write)
    ftp.quit()
    return local_path


def wget_file(url, local_path):
    """
    If local has wget, this can be used.

    Arguments
    ---------
    url : str
        Path on ftp server
    local_path : str
        Path to save file (usually url without ftp://)

    Returns
    -------
    local_path : str
        local_path
    """
    import os
    if not os.path.exists(local_path):
        cmd = f'wget -r -N {url}'
        os.system(cmd)

    return local_path


def request_file(url, local_path):
    """
    Only works with http and https

    Arguments
    ---------
    url : str
        Path on ftp server
    local_path : str
        Path to save file (usually url without https://)

    Returns
    -------
    local_path : str
        local_path
    """
    import requests
    import shutil
    import os

    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_path


def read_netrc(netrcpath, server):
    import netrc
    nf = netrc.netrc(netrcpath)
    return nf.authenticators(server)


def mpestats(df, refkey='obs'):
    """
    Calculate typical model statistics
    [1] https://epa.gov/sites/production/files/2015-11/
        modelperformancestatisticsdefinitions.docx

    Example assuming you have a csv with 'obs' and 'mod' columns:

    ```
    import pandas as pd

    df = pd.read_csv('obs_mod.csv')
    statdf = getstats(df)
    statdf.to_csv('stats.csv')
    ```

    Arguments
    ---------
    df : pandas.DataFrame
        Each column should be the reference observation or an estimate
        of the reference.
    refkey : str
        Column to be used as a reference observation. All other columns are
        will be compared to refkey

    Returns
    -------
    mpedf : pandas.DataFrame
        DataFrame with statistics by estimate.
        - Descriptive statistics :
          - count, mean, std, min, 5%, 25%, 50%, 75%, 95%, max
          - skew : 50% / mean
          - cov : std / mean
        - Evaluation statistics :
          - r : Pearson Correlation
          - mb : mean bias mean(yhat - yref)
          - me : mean bias mean(|yhat - yref|)
          - rmse : Root Mean Square Error mean((yhat - yref)**2)**0.5
          - nmb : mb / mean(yref) * 100 (as %)
          - nme : me / mean(yref) * 100 (as %)
          - fmb : 200 * mb / (mean(yref) + mean(y)) (as %)
          - fme : 200 * me / (mean(yref) + mean(y)) (as %)
          - ioa : 1 - sum((yhat - yref)**2) / sum(
                        (|yhat - mean(yref)| + |yref - mean(yref)|)**2
                  )

    """
    sdf = df.describe().T
    sdf['5%'] = df.quantile(0.05)
    sdf['95%'] = df.quantile(0.95)
    dks = [
        'count', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max'
    ]
    sdf = sdf[dks].copy()
    om = sdf.loc[refkey, 'mean']
    bias = df.subtract(df[refkey], axis=0)
    minusom = df.subtract(om, axis=0).abs()
    ioaden = (
        minusom.add(minusom[refkey], axis=0)**2
    ).sum()
    se = bias**2
    sse = se.sum()
    sdf['skew'] = sdf['50%'] / sdf['mean']
    sdf['cov'] = sdf['std'] / sdf['mean']
    sdf['r'] = df.corr()[refkey]
    sdf['mb'] = bias.mean().T
    sdf['me'] = bias.abs().mean().T
    sdf['rmse'] = se.mean()**.5
    sdf['nmb'] = sdf['mb'] / om * 100
    sdf['nme'] = sdf['me'] / om * 100
    sdf['fmb'] = sdf['mb'] / (om + sdf['mean']) * 200
    sdf['fme'] = sdf['me'] / (om + sdf['mean']) * 200
    sdf['ioa'] = 1 - sse / ioaden
    sdf.index.name = 'key'

    return sdf


def to_geopandas(
    x, y, z, crs, edges=None, colors=None, names=None,
    under='#808080', over='#000000'
):
    """
    Converts z into a set of polygons that are returned as a geopandas
    GeoDataFrame

    Inspired by
    http://geoexamples.blogspot.com/2013/08/creating-vectorial-isobands-with-
      python.html

    Arguments
    ---------
    x : array-like
        1-d x-coordinates in crs units
    y : array-like
        1-d y-coordinates in crs units
    z : array-like
        2-d (ny,nx) values at the y/x coordinates
    crs : str
        Projection string (PROJ4 or anything geopandas compatible)
    edges : list
        List of numerical boundaries for norm and cmap
    colors : list
        List of colors to be used for cmap
    names : list
        Names of intervals (one fewer than edges)
    under : str
        If None, do not automatically add an under category.
        If not None, add a category (z.min(), edges[0]) with color=under
    over : str
        If None, do not automatically add an over category.
        If not None, add a category (z.max(), edges[-1]) with color=over

    Returns
    -------
    gdf, cmap, norm : geopandas.GeoDataFrame, matplotlib.cmap, matplotlib.norm
        Contains 1 row for each interval between edges, including rows with
        empty Polygons
    """
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from shapely.geometry import Polygon, MultiPolygon, box
    import geopandas as gpd

    if edges is None:
        # Based on PM from DMC
        edges = [-5., 10, 20, 30, 50, 70, 90, 120, 500.4]
    zmin = z.min() - 1e-20
    zmax = z.max() + 1e-20
    if colors is None:
        colors = [
                '#009600', '#99cc00', '#ffff99', '#ffff00', '#ffcc00',
                '#f79900', '#ff0000', '#d60093',
        ][:len(edges) - 1]

    colors = [c for c in colors]
    if zmin < edges[0] and under is not None:
        colors.insert(0, under)
        edges = np.append(zmin, edges)

    if zmax > edges[-1] and over is not None:
        colors.append(over)
        edges = np.append(edges, zmax)

    if names is None:
        names = [
            f'{start} to {end:.4g}'
            for start, end in zip(edges[:-1], edges[1:])
        ]

    centers = np.interp(
        np.arange(len(edges) - 1) + 0.5, np.arange(len(edges)), np.array(edges)
    )
    cmap, norm = plt.matplotlib.colors.from_levels_and_colors(
        np.array(edges) + 0.0, colors, extend='neither'
    )
    fig, ax = plt.subplots(1, 1, dpi=300)

    # Clip the top and bottom of the scale
    Z = np.ma.maximum(np.ma.minimum(np.ma.masked_invalid(z), 500), 0)

    qcs = ax.contourf(
        x, y, Z, levels=edges, cmap=cmap, norm=norm
    )
    mpolys = []
    i = 0
    # qcs.collections has been deprecated. On inspection,
    # qcs.collections[i].get_paths() is identical to [qcs.get_paths()[i]]
    # as a result, I have changed the setup to use get_paths
    # https://github.com/matplotlib/matplotlib/blob/v3.8.1/lib/matplotlib
    #  /contour.py#L987
    try:
        paths = qcs.get_paths()
        pathgroups = [[p] for p in paths]
    except Exception:
        pathgroups = [c.get_paths() for c in qcs.collections]

    assert len(pathgroups) == len(centers)
    for pi, pathg in enumerate(pathgroups):
        polys = []
        for ppi, path in enumerate(pathg):
            if path.codes is None:
                continue
            nbadpoly = []
            rings = []
            xys = None
            for xy, c in zip(path.vertices, path.codes):
                if c == 1:
                    if xys is not None:
                        rings.append(xys)
                        i += 1
                    xys = [xy]
                else:
                    xys.append(xy)

            rings.append(xys)

            nr = len(rings)
            if nr > 0:
                try:
                    rings = [r for r in rings if len(r) > 3]
                    poly = Polygon(rings[0], rings[1:])
                    polys.append(poly)
                except Exception as e:
                    nbadpoly.append(e)
            if len(nbadpoly) > 0:
                warnings.warn(
                    f'*Lost {len(nbadpoly)} poly for {names[pi]}: {nbadpoly}'
                )
        mpolys.append(dict(
            Name=names[pi], AQIC=centers[pi], geometry=MultiPolygon(polys),
            OGR_STYLE=f'BRUSH(fc:{colors[pi]})',
        ))

    if len(mpolys) == 0:
        gdf = gpd.GeoDataFrame([
                dict(Name='BLANK', AQIC=-999, OGR_STYLE='BRUSH(fc:#808080')
            ], geometry=[box(-130, 20, -129.999, 20.001)], crs=4326
        ).to_crs(crs)
    else:
        df = pd.DataFrame(mpolys)
        gdf = gpd.GeoDataFrame(
            df.drop('geometry', axis='columns'), geometry=df['geometry'],
            crs=crs
        )

    plt.close(fig)
    return gdf, cmap, norm


def to_geojson(
    outpath, *args, simplify=.01, precision=5, outcrs=4326, description=None,
    **kwds
):
    """
    Thin wrapper around to_geopandas with addition arguments. See to_geopandas
    for definition of other arguments (x, y, z, colors, edges, names, over,
    under).

    Arguments
    ---------
    outpath : str
        Path to save the geojson to
    simplify : float
        Level of simplification that occurs in long/lat space.
    precision : int
        Level of precision to hold in output coordinate.
    description : str
        If not None, add DESCRIPTON to driver_options
    kwds: mappable
        Passed to to_geopandas

    Returns
    -------
    None
    """
    from shapely import wkt
    import logging

    gdf, cmap, norm = to_geopandas(*args, **kwds)
    verbose = kwds.get('verbose', 0)
    if outcrs is not None:
        gdf = gdf.to_crs(outcrs)
    if precision is not None:
        if verbose > 0:
            logging.info('Reducing precision of coordinates')
        gdf['geometry'] = gdf.geometry.apply(
            lambda x: wkt.loads(wkt.dumps(x, rounding_precision=precision))
        )

    if simplify is not None:
        if verbose > 0:
            logging.info('Simplify')
        gdf['geometry'] = gdf['geometry'].simplify(simplify)
    driver_opts = {
        'driver': 'GeoJSON',
        'COORDINATE_PRECISION': 7
    }
    if description is not None:
        driver_opts['DESCRIPTION'] = description

    gdf.to_file(outpath, **driver_opts)


def df2nc(
    tgtdf, varattrs, fileattrs, coordkeys=None, units='unknown', outpath=None
):
    """
    Converts gridded target dataframe to a gridded Dataset
    Arguments
    ---------
    tgtdf: pandas.DataFrame
        DataFrame with gridded predictions
    varattrs: dict
        Dictionary of attributes for each variable that should be retained.
    fileattrs: dict
        File attributes to document the results.
    coordkeys: list
        Names of coordinates (defaults to ['time', 'y', 'x'])

    Returns
    -------
    tgtds : xr.Dataset
        Dataset with variables and properties defined above.
    """
    import xarray as xr
    import pandas as pd
    import pyproj
    from datetime import datetime

    if coordkeys is None:
        coordkeys = ['time', 'y', 'x']
    keepkeys = [k for k in varattrs if k in tgtdf.columns]
    tgtds = tgtdf[coordkeys + keepkeys].set_index(coordkeys).to_xarray()
    tgtds.coords['time'] = pd.to_datetime(tgtds.coords['time'])

    for k in tgtds.data_vars:
        tgtds[k] = tgtds[k].astype('f')
        tgtds[k].attrs.setdefault('long_name', k)
        tgtds[k].attrs.setdefault('units', units)
        tgtds[k].attrs.update(varattrs[k])

    tgtds.attrs.update(fileattrs)
    now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S%z')
    if 'crs_proj4' in fileattrs:
        proj = pyproj.Proj(fileattrs['crs_proj4'])
        cfattrs = proj.crs.to_cf()
        # the parameters below are required for Panoply to work correctly.
        tgtds['crs'] = xr.DataArray(0, dims=(), attrs=cfattrs)
        # Not sure why, but Panoply requires this attribute
        tgtds['crs'].attrs['latitude_of_projection_origin'] = (
            tgtds['crs'].attrs['standard_parallel']
        )
        # Projected coordinate for NAQFC is in km
        tgtds.coords['x'].attrs.update(
            units='km',
            standard_name='projection_x_coordinate',
        )
        tgtds.coords['y'].attrs.update(
            units='km',
            standard_name='projection_y_coordinate',
        )
        tgtds.attrs['Conventions'] = 'CF-1.6'
    tgtds.attrs.setdefault('creation_date', now)
    if outpath is not None:
        tgtds.to_netcdf(outpath)
    return tgtds


def to_webpng(
    ncpath, bbox=None, dx=2000., dy=2000., key='FUSED_aVNA', pngpath=None
):
    """
    Convert AirFuse netcdf output to web mercator PNG for use with online
    mapping tools.

    Arguments
    ---------
    ncpath : str or xarray.Dataset
        If str, open Dataset from path.
        Otherwise, use ncpath as Dataset
        Either must have key in data_vars and crs_proj4 in attrs
    bbox : tuple
        Approximate bounding box in WGS 84 lllon, lllat, urlon, urlat.
        Actually upper right coordinates may slightly exceed. The true extent
        is provided as a property of the png. default: -153, 10, -49, 62
    dx : float
        Web Mercator width in meters of pixel in output png
    dy : float
        Web Mercator height in meters of pixel in output png
    key : str
        Variable from ncpath to use (default: FUSED_aVNA)
    pngpath : str
        Path to save output as a png.

    Returns
    ---------
    img : PIL.Image
        Output image as a pillow Image object
    """
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import airfuse.style
    import pyproj
    import xarray as xr
    import numpy as np

    norm = airfuse.style.ant_1hpm_norm
    cmap = airfuse.style.ant_1hpm_cmap
    if isinstance(ncpath, str):
        f = xr.open_dataset(ncpath)
    else:
        f = ncpath
        ncpath = 'object provided'

    proj = pyproj.Proj(f.crs_proj4)
    wmproj = pyproj.Proj(3857)
    if bbox is None:
        bbox = [-153., 10., -49., 62.]  # consider defining bbox based on f

    bbox = np.asarray(bbox)
    wmx, wmy = wmproj(bbox[[0, 2]], bbox[[1, 3]])
    dx = dy = 2000  # 2km resolution in Web Mercator Space
    nx = int((wmx[1] - wmx[0]) // dx) + 1  # Ensure full coverage
    ny = int((wmy[1] - wmy[0]) // dy) + 1
    wmxb = wmx[0], wmx[0] + nx * 2000  # redefine bounds with extra cell
    wmyb = wmy[0], wmy[0] + ny * 2000
    # define centroids in Web Mercator Space
    wmx = np.linspace(wmxb[0] + dx / 2, wmxb[1] - dx / 2, nx)
    wmy = np.linspace(wmyb[0] + dy / 2, wmyb[1] - dy / 2, ny)
    wmX, wmY = np.meshgrid(wmx, wmy)
    # Convert Web Mercator centroids and bounds to lon/lat
    LON, LAT = wmproj(wmX, wmY, inverse=True)
    wmbbox = np.array([wmxb, wmyb]).T.ravel()
    gbbox = np.array(wmproj(wmxb, wmyb, inverse=True)).T.ravel()
    lX, lY = proj(LON, LAT)
    lX = xr.DataArray(lX, dims=('lat', 'lon'))
    lY = xr.DataArray(lY, dims=('lat', 'lon'))
    # Interpolate to lon/lat centers
    lZ = f[key].interp(x=lX, y=lY, method='linear')
    a = cmap(norm(lZ[0, ::-1]), bytes=True)
    a[:, :, 3] *= ~lZ[0, ::-1].isnull()
    img = Image.fromarray(a, mode='RGBA')
    img = img.convert("P", palette=Image.WEB, colors=256)
    metadata = PngInfo()
    metadata.add_text("title", f.title)
    metadata.add_text("author", f.author)
    metadata.add_text("institution", f.institution)
    desc = (
        f'{key} Bilinearly interpolated to EPSG:3857 from {ncpath}\n'
        + f' that was created on {f.creation_date}. Original metadata'
        + ' below:\n\n' + f.description
    )
    tstr = f.time.dt.strftime('%FT%H:%MZ').values[0]
    gdesc = (
        "crs (EPSG:3857) is the coordinate reference system and extent defines"
        + " the bounding box (lllon, lllat, urlon, urlat) in WGS84 degrees."
        + " wm_extent defines the bounding box in crs (i.e., Web Mercator)."
    )
    img.info.update({
        "description": desc, "date_time": tstr,
        "georeference_description": gdesc, "crs": 'ESPG:3857',
        "extent": str(gbbox), "wm_extent": str(wmbbox)
    })
    for k, v in img.info.items():
        metadata.add_text(k, v)

    if pngpath is not None:
        img.save(pngpath, pnginfo=metadata)

    return img
