def to_geopandas(x, y, z, crs, edges, colors, labels=None, empty=True):
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
    edges : array-like
        Color bin edges (n+1); use -inf and inf to enable over/under categories
    colors : array-like
        Colors (n) names or hex codes for the color of each bin.
    labels : array-like
        Labels (n) of intervals
    empty : bool
        If True (default), keep empty polygons

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Contains 1 row for each interval between edges, including rows with
        empty Polygons
    """
    import geopandas as gpd
    import pandas as pd
    from shapely import GeometryType, from_ragged_array, unary_union, wkt
    from shapely.geometry import box
    from contourpy import contour_generator
    import logging
    import matplotlib.colors as mc
    logger = logging.getLogger('airfuse.utils.to_geopandas')
    minv = float(z.min())
    maxv = float(z.max())
    inf = float('inf')
    nc = len(colors)
    assert (nc + 1) == len(edges)
    if labels is not None:
        assert nc == len(labels)
    else:
        labels = []
        for i in range(nc):
            lo = edges[i]
            hi = edges[i + 1]
            if lo == float('-inf'):
                labels.append(f'<{hi}')
            elif hi == float('inf'):
                labels.append(f'>={lo}')
            else:
                labels.append(f'{lo} <= z < {hi}')

    # Each contour color is represented by a multipolygon
    mpolys = []
    for i in range(nc):
        mylbl = labels[i]
        if edges[i] == -inf:
            lower = minv
        else:
            lower = edges[i]
        if edges[i + 1] == inf:
            upper = maxv
        else:
            upper = edges[i + 1]
        mycolor = mc.to_hex(colors[i])
        try:
            cont_gen = contour_generator(
                z=z, x=x, y=y, fill_type="ChunkCombinedOffsetOffset"
            )
            # Chunk combined offset offset has only on set of values for
            # three categories points, offsets, outer_offsets
            pts, offs, outoffs = cont_gen.filled(
                edges[i], edges[i + 1]
            )
            # When there is no polygon, the points element is None
            if pts[0] is None:
                logger.info(f'No polygon for {mylbl}')
                if empty:
                    mply = wkt.loads('POLYGON EMPTY')
                else:
                    continue
            else:
                # pts, offs, outoffs are for shapely's from_ragged_array func
                ropts = GeometryType.POLYGON, pts[0], (offs[0], outoffs[0])
                polygons = from_ragged_array(*ropts)
                # The resulting polygons are combined into a multipolygon
                mply = unary_union(polygons)
            # And stored with metadata for the geopandas.GeoDataFrame
            mpolys.append(dict(
                label=mylbl, geometry=mply, OGR_STYLE=f'BRUSH(fc:{mycolor})',
                lower=lower, upper=upper, color=mycolor,
            ))
        except Exception as e:
            logger.warning(f'*Lost polygon for {mylbl}: {str(e)}')

    if len(mpolys) == 0:
        nan = float('nan')
        mycolor = '#808080'
        gdf = gpd.GeoDataFrame([
            dict(
                label='BLANK', OGR_STYLE=f'BRUSH(fc:{mycolor})',
                lower=nan, upper=nan, color=mycolor
            )
        ], geometry=[box(x.min(), y.min(), x.max(), y.max())], crs=crs)
    else:
        df = pd.DataFrame(mpolys)
        logger.info('Geometry Summary:')
        lstr = repr(df.drop('geometry', axis='columns'))
        for log in lstr.split('\n'):
            logger.info(log)
        gdf = gpd.GeoDataFrame(
            df.drop('geometry', axis='columns'), geometry=df['geometry'],
            crs=crs
        )
    return gdf


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

    gdf = to_geopandas(*args, **kwds)
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
