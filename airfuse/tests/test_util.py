import pytest


def test_mpestats():
    from ..util import mpestats
    import numpy as np
    import pandas as pd
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = np.sin(x)
    err = np.random.normal(size=x.size)
    df = pd.DataFrame(dict(mod=y + err, perfect=y, obs=y), index=x)
    mpedf = mpestats(df)
    assert mpedf.loc['obs', 'r'] == 1
    assert mpedf.loc['perfect', 'r'] == 1
    assert mpedf.loc['perfect', 'me'] == 0
    assert mpedf.loc['mod', 'r'] < 1
    assert mpedf.loc['mod', 'me'] > 0


@pytest.mark.xfail(strict=False, reason='Requires geopandas')
def test_to_geopandas():
    import numpy as np
    from ..util import to_geopandas
    x = np.arange(0, 5)
    y = np.arange(0, 2)
    Z = np.arange(10).reshape(2, 5)
    edges = np.arange(9) + 0.5
    colors = [
        '#009600', '#99cc00', '#ffff99', '#ffff00', '#ffcc00',
        '#f79900', '#ff0000', '#d60093',
    ]
    gdf, cmap, norm = to_geopandas(
        x, y, Z, crs=4326, edges=edges, colors=colors,
        under='#808080', over='#000000'
    )
    assert (norm.boundaries[1:-1] == edges).all()
    assert gdf.shape[0] == Z.size
    ustyles = gdf['OGR_STYLE'].unique()
    for c in colors:
        assert f'BRUSH(fc:{c})' in ustyles


@pytest.mark.xfail(strict=False, reason='Requires geopandas')
def test_to_geojson():
    import os
    import numpy as np
    import geopandas as gpd
    from ..util import to_geopandas, to_geojson
    import tempfile
    x = np.arange(0, 5)
    y = np.arange(0, 2)
    Z = np.arange(10).reshape(2, 5)
    edges = np.arange(9) + 0.5
    colors = [
        '#009600', '#99cc00', '#ffff99', '#ffff00', '#ffcc00',
        '#f79900', '#ff0000', '#d60093',
    ]
    gdf, cmap, norm = to_geopandas(
        x, y, Z, crs=4326, edges=edges, colors=colors,
        under='#808080', over='#000000'
    )
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        to_geojson(
            tf.name, x, y, Z, crs=4326, edges=edges, colors=colors,
            under='#808080', over='#000000'
        )
        gdf2 = gpd.read_file(tf.name)
        # Ensure that the temporary file unlinked
        if tf:
            os.unlink(tf.name)
    assert (gdf2 == gdf).drop('geometry', axis=1).all().all()
