import pathlib
import numpy as np
import pytest

from reader.mrbfile import MRBFile
from reader.tagged_data import LabelData, RawData
from reader.segment_info import SegmentInfo
from reader.templates import template



class LabelDataMock(LabelData):
    """
    Mock class that should only overwrite the default constructor/initializer
    for easier local test setup without a metadata dictionary. 
    """
    def __init__(self, segmentinfos, data=None):
        self.infos = {
            si.label_value : si for si in segmentinfos 
        }
        self.data = data
    

    @property
    def metadata(self):
        return super().metadata
    


@pytest.fixture
def mrbfile():
    fpath = pathlib.Path(
        'C:/Users/Jannik/Desktop/mrbreader/tests/assets/testmrb_multiseg.mrb')
    mrb = MRBFile(fpath)
    return mrb


@pytest.fixture
def mock_raw_data():
    target_shape = (10, 10, 10)
    mean = 0
    sigma = 1
    raw_data = np.random.default_rng().normal(loc=mean,
                                              scale=sigma,
                                              size=target_shape)
    return raw_data


@pytest.fixture
def mock_label_data():
    """
    Mock label data of shape (10, 10, 10) and values in [0, 1, 2, 3]
    """
    # use mock data with reduced spatial size: we want efficient tests
    target_shape = (10, 10, 10)
    # add 0 for background emulation
    label_values = [0, 1, 2, 3]

    mock_label_volume = np.random.default_rng().choice(
        label_values, size=np.prod(target_shape)
    )
    # guarantee the existence of every label value by sampling
    # three indices per label value that are deterministically
    # given that label value
    fixed_idcs = np.random.default_rng().choice(
        np.prod(target_shape), size=(len(label_values), 3),
        replace=False
    )
    for idx, label_value in enumerate(label_values):
        mock_label_volume[fixed_idcs[idx, :]] = label_value

    return mock_label_volume.reshape(target_shape)


@pytest.fixture
def mock_tagged_raw_data(mock_raw_data):
    metadata = {
        'attr_1' : 'This is test metadata',
        'attr_2' : 123,
        'attr_3' : 3.141,
        'attr_4' : 'previous k-v-paris should be int and float'
    }
    return RawData(data=mock_raw_data, metadata=metadata)


@pytest.fixture
def segmentation(mrbfile, mock_label_data):
    """
    Set up segmentation mock data
    """
    seg = mrbfile.read_segmentations()[0]
    seg.data = mock_label_data
    # sanity checking - remove later
    assert seg.data.shape == (10, 10, 10)
    assert set(np.unique(seg.data)) == set((0, 1, 2, 3))

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
         'bogen', 'Bogengnge')
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



@pytest.fixture
def synthetic_segmentation(mock_label_data):
    """
    Fully synthetic LabelData object.
    Constructed using the local `LabelDataMock` class. 
    """
    background_si = SegmentInfo(
        'background', (0, 0, 0), 'Segment_0', 0,
        (1, 2, 3, 4, 5, 6), 'none', 'none', 'none', 'none'
    )
    cochlea_si = SegmentInfo(
        'Cochlea', (1, 0, 0), 'Segment_1', 1,
        (1, 2, 3, 4, 5, 6), 'none', 'none', 'none', 'none'
    )
    vestibulum_si = SegmentInfo(
        'Vestibulum', (0, 1, 0), 'Segment_2', 2,
        (1, 2, 3, 4, 5, 6), 'none', 'none', 'none', 'none'
    )
    bogen_si = SegmentInfo(
        'Bogen', (0, 0, 1), 'Segment_3', 3,
        (1, 2, 3, 4, 5, 6), 'none', 'none', 'none', 'none'
    )
    seginfos = [background_si, cochlea_si,
                vestibulum_si, bogen_si]

    segmentation = LabelDataMock(
        seginfos, data=mock_label_data)
    
    return segmentation
