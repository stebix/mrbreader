import pathlib
import numpy as np
import pytest

from reader.mrbfile import MRBFile
from reader.tagged_data import SegmentationData, SegmentInfo, RawData

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
    # reduce spatial dims: we want efficient tests
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


def test_label_values_data_consistency(segmentation):
    data_label_values = set(np.unique(segmentation.data))
    # match infos dict keys to unique elements in data array
    assert set(segmentation.infos.keys()) == set(np.unique(segmentation.data))
    # match SegmentInfo label_value attributes to unique elements in data array
    assert set(si.label_value for si in segmentation.infos.values()) == data_label_values 


def test_relabel_effect_on_data(segmentation):
    original_data = np.copy(segmentation.data)
    old_label = np.unique(original_data)[-1]
    # relabel to artificial number
    new_label = 1701
    segmentation.relabel(old_label, new_label)
    new_data = segmentation.data
    assert new_data[original_data == old_label] == new_label



