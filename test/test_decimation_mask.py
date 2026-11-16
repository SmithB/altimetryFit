import numpy as np
import pytest
import pointCollection as pc
from altimetryFit.decimation_mask import in_no_decimation_mask, blockmedian_outside_mask, read_no_decimation_mask

rng = np.random.default_rng(1)


def make_data(N=5000):
    return pc.data().from_dict({'x': rng.uniform(-1000, 1000, N), 'y': rng.uniform(-1000, 1000, N),
                                'z': rng.normal(size=N), 'time': rng.uniform(0, 1, N)})


def xyz_set(D):
    return set(zip(np.round(D.x, 6), np.round(D.y, 6), np.round(D.z, 6)))


xx = np.arange(-1500, 1501, 50.)
mask_2d = pc.grid.data().from_dict({'x': xx, 'y': xx, 'z': (xx[None, :] < 0)*np.ones((xx.size, 1))})


def test_no_mask_is_blockmedian():
    D = make_data()
    D1 = D.copy()
    D.blockmedian(100.)
    blockmedian_outside_mask(D1, 100., None)
    assert np.array_equal(D.x, D1.x) and np.array_equal(D.z, D1.z)


def test_2d_mask():
    D = make_data()
    # the mask nodes are 1 for x < 0, so the interpolated mask crosses 0.5 at x=-25
    inside = D.x < -25
    D_in = D.copy_subset(np.flatnonzero(inside))
    D_out = D.copy_subset(np.flatnonzero(~inside)).blockmedian(100.)
    out = blockmedian_outside_mask(D.copy(), 100., mask_2d)
    # every point inside the mask is kept, and the outside points are their blockmedian
    assert np.array_equal(in_no_decimation_mask(mask_2d, D.x, D.y), inside)
    assert xyz_set(out) == xyz_set(D_in) | xyz_set(D_out)
    assert out.size == D_in.size + D_out.size
    # a mask covering everything keeps all the data
    all_in = pc.grid.data().from_dict({'x': xx, 'y': xx, 'z': np.ones((xx.size, xx.size))})
    assert blockmedian_outside_mask(D.copy(), 100., all_in).size == D.size


def test_3d_mask_and_shapes():
    # a mask whose edge moves with time: inside where x < -200 + 400 t.  Between
    # epochs, the mask is interpolated linearly, so the epochs must be close enough
    # that the edge moves less than the tolerance below between them
    t = np.linspace(0, 1, 21)
    z = (xx[None, :, None] < -200 + 400*t[None, None, :])*np.ones((xx.size, 1, 1))
    mask_3d = pc.grid.data().from_dict({'x': xx, 'y': xx, 't': t, 'z': z.astype(float)})
    D = make_data()
    expected = D.x < -200 + 400*D.time
    got = in_no_decimation_mask(mask_3d, D.x, D.y, D.time)
    # bilinear interpolation of the 0/1 mask is exact away from its edge
    far = np.abs(D.x - (-200 + 400*D.time)) > 60
    assert np.array_equal(got[far], expected[far])
    # the output has the shape of the input coordinates
    assert in_no_decimation_mask(mask_2d, D.x.reshape(-1, 2), D.y.reshape(-1, 2)).shape == (D.size//2, 2)
    with pytest.raises(ValueError):
        in_no_decimation_mask(mask_3d, D.x, D.y)


def test_read_mask(tmp_path):
    mask_2d.to_h5(str(tmp_path/'mask.h5'))
    m = read_no_decimation_mask(str(tmp_path/'mask.h5'), [0, 0], {'x': 1000, 'y': 1000}, pad=100)
    assert m.x.min() >= -600 and m.x.max() <= 600
    assert read_no_decimation_mask(None, [0, 0], {'x': 1000, 'y': 1000}) is None
    assert read_no_decimation_mask(mask_2d, [0, 0], {'x': 1000, 'y': 1000}) is mask_2d
