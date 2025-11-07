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
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from shapely.geometry import Polygon, MultiPolygon, box
    import geopandas as gpd
    import logging
    logger = logging.getLogger('airfuse.utils.to_geopandas')

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
                logger.warn(
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
