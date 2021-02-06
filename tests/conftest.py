import pathlib
import numpy as np
import pytest

from reader.mrbfile import MRBFile


@pytest.fixture
def mrbfile():
    fpath = pathlib.Path(
        'C:/Users/Jannik/Desktop/mrbreader/tests/assets/testmrb_multiseg.mrb')
    mrb = MRBFile(fpath)
    return mrb


@pytest.fixture
def segmentation(mrbfile):
    """
    Set up segmentation mock data
    """
    seg = mrbfile.read_segmentations()[0]
    label_candidates = list(seg.infos.keys())
    # use mock data with reduced spatial size: we want efficient tests
    target_shape = (10, 10, 10)
    downsampled_data = np.random.default_rng().choice(
        label_candidates, size=np.prod(target_shape)
    )
    # guarantee the existence of every label value by sampling
    # three indices per label value that are deterministically
    # given that label value
    fixed_idcs = np.random.default_rng().choice(
        np.prod(target_shape), size=(len(label_candidates), 3),
        replace=False
    )
    for idx, label_candidate in enumerate(label_candidates):
        downsampled_data[fixed_idcs[idx, :]] = label_candidate

    seg.data = downsampled_data.reshape(target_shape)

    # sanity checking - remove later
    assert seg.data.shape == target_shape
    assert set(np.unique(seg.data)) == set(label_candidates)

    return seg