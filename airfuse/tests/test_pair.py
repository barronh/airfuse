import os
import pytest
_bbox = (-80, 35, -55, 50)
_haspakey = os.path.exists(os.path.expanduser('~/.purpleairkey'))


def test_pair_rsigairnow():
    import tempfile
    from ..points import rsigairnow
    from ..layers import naqfc
    date = '2025-07-01'
    with tempfile.TemporaryDirectory() as td:
        opts = dict(spc='pm25', nowcast=False, inroot=td)
        an = rsigairnow(**opts)
        lay = naqfc(**opts)
        modvar = lay.get(date)
        padf = an.pair(date, modvar, proj=lay.proj)
        assert (padf.shape[0] > 0)
        assert ('obs' in padf.columns)
        assert ('mod' in padf.columns)
        assert ('x' in padf.columns)
        assert ('y' in padf.columns)


@pytest.mark.skipif(not _haspakey, reason="requires ~/.purpleairkey")
def test_pair_rsigpurpleair():
    import tempfile
    from ..points import rsigpurpleair
    from ..layers import naqfc
    date = '2025-07-01'
    with tempfile.TemporaryDirectory() as td:
        opts = dict(spc='pm25', nowcast=False, inroot=td)
        an = rsigpurpleair(**opts)
        lay = naqfc(**opts)
        modvar = lay.get(date)
        padf = an.pair(date, modvar, proj=lay.proj)
        assert (padf.shape[0] > 0)
        assert ('obs' in padf.columns)
        assert ('mod' in padf.columns)
        assert ('x' in padf.columns)
        assert ('y' in padf.columns)
