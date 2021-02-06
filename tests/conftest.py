import pathlib
import numpy as np
import pytest

from reader.mrbfile import MRBFile
from reader.templates import template


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
    # add 0 for background emulation
    label_candidates = list(seg.infos.keys()) + [0]
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


@pytest.fixture
def standard_template():
    return template


@pytest.fixture
def fullchange_template():
    cochlea_equivalents = frozenset(
    ('cochlea', 'Cochlea', 'chl', 'Schnecke', 'schnecke')
    )
    vestibulum_equivalents = frozenset(
        ('vestibulum', 'Vestibulum', 'vest')
    )
    canals_equivalents = frozenset(
        ('Bogengänge', 'bogengänge', 'bogengaenge', 'Bogengaenge',
         'canals', 'Canals', 'semicircular canals', 'Bogen',
         'bogen')
    )
    template = {
        cochlea_equivalents : {
            'name' : 'foo',
            'color' : (1, 0, 0),
            'ID' : 'Segment_foo',
            'label_value' : 10
        },
        vestibulum_equivalents : {
            'name' : 'bar',
            'color' : (0, 1, 0),
            'ID' : 'Segment_bar',
            'label_value' : 20
        },
        canals_equivalents : {
            'name' : 'baz',
            'color' : (0, 0, 1),
            'ID' : 'Segment_baz',
            'label_value' : 30
        }
    }
    return template