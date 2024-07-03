import pytest


def test_get_constant():
    from ..mod.constant import get_constant
    from ..mod import get_model
    date = '2024-06-01'
    ones = get_model(date, key='o3', model='NULL', verbose=0) + 1
    twos = get_constant(date, bbox=(-85, 35, 65, 50), default=2)
    assert (twos / ones).mean() == 2
    assert ones.size > twos.size


@pytest.mark.xfail
def test_get_bad():
    from ..mod import get_model
    get_model('2024-06-01', key='o3', model='oops', verbose=0)


@pytest.mark.xfail
def test_get_goesfail():
    from ..mod import get_model
    get_model('2024-06-01', key='o3', model='GOES')
