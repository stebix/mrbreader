import numpy as np
import scipy.ndimage as ndimage
import pytest

from reader.rescaler import Rescaler


def test_write_only_attributes():
    ovs = 1
    rvs = 2
    intorder = 1
    rescaler = Rescaler(original_voxel_size=ovs,
                        rescaled_voxel_size=rvs,
                        interpolation_order=intorder)
    
    assert rescaler.original_voxel_size == ovs
    assert rescaler.rescaled_voxel_size == rvs
    assert rescaler.interpolation_order == intorder

    with pytest.raises(AttributeError):
        rescaler.original_voxel_size = 137


def test_rescale_volume():
    ovs = 2
    rvs = 1
    intorder = 5
    rescaler = Rescaler(original_voxel_size=ovs,
                        rescaled_voxel_size=rvs,
                        interpolation_order=intorder)

    original_volume = np.random.default_rng().normal(size=(20, 20, 20))
    rescaler_result = rescaler.rescale_volume(original_volume)
    # test plausibility of output via zoom function
    expected_volume = ndimage.zoom(original_volume, zoom=2,
                                   order=intorder, grid_mode=False)

    import matplotlib.pyplot as plt
    slc = np.s_[20, ...]
    fig, axes = plt.subplots(ncols=2, nrows=2)
    axes = axes.flat
    ax = axes[0]
    ax.imshow(rescaler_result[slc])
    ax.set_title(f'rescaler, shp {rescaler_result.shape}')

    ax = axes[1]
    ax.imshow(expected_volume[slc])
    ax.set_title(f'expected, shp {expected_volume.shape}')

    ax = axes[2]
    ax.imshow(np.abs(expected_volume[slc] - rescaler_result[slc]))
    ax.set_title('deviation')

    ax = axes[3]
    ax.imshow(original_volume[10, ...])
    ax.set_title('original')



    plt.show()

    totdev = np.sum(np.abs(expected_volume - rescaler_result))
    print(f'totdev: {totdev}')



    # assert np.allclose(rescaler_result, expected_volume)