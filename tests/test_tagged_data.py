import copy
import numpy as np
import pytest

from reader.mrbfile import MRBFile
from reader.tagged_data import SegmentationData


def equal_seginfos(instance_a, instance_b):
    """
    helper function to quickly compare two
    `reader.tagged_data.SegmentInfo` instances
    """
    # check equality for the following attribute names
    relevant_attrs = ['name', 'color', 'ID', 'extent', 'layer', 'tags']
    for relattr in relevant_attrs:
        if not getattr(instance_a, relattr) == getattr(instance_b, relattr):
            return False
    else:
        return True


def test_label_values_data_consistency(segmentation):
    data_label_values = set(np.unique(segmentation.data))
    # match infos dict keys to unique elements in data array
    assert set(segmentation.infos.keys()) == set(np.unique(segmentation.data))
    # match SegmentInfo label_value attributes to unique elements in data array
    assert set(si.label_value for si in segmentation.infos.values()) == data_label_values 


def test_relabel_effect_on_infos(segmentation):
    old_label = 1
    old_seginfo = copy.deepcopy(segmentation.infos[old_label])
    new_label = 1701
    segmentation.relabel(old_label, new_label)
    new_seginfo = segmentation.infos[new_label]
    assert equal_seginfos(old_seginfo, new_seginfo)
    

def test_relabel_effect_on_data(segmentation):
    original_data = np.copy(segmentation.data)
    old_label = np.unique(original_data)[-1]
    # relabel to artificial number
    new_label = 1701
    segmentation.relabel(old_label, new_label)
    new_data = segmentation.data
    assert np.all(new_data[original_data == old_label] == new_label)
    assert np.all(new_data[original_data != old_label] != new_label)


def test_swaplabel_effect_on_data(segmentation):
    label_a = 1
    label_b = 2
    original_data = np.copy(segmentation.data)
    info_a = segmentation.infos[label_a]
    info_b = segmentation.infos[label_b]
    segmentation.swaplabel(label_a, label_b)
    new_data = segmentation.data
    # where label_a was, now should be label_b and vice versa
    assert np.all(new_data[original_data == label_a] == label_b)
    assert np.all(new_data[original_data == label_b] == label_a)
    # reversed case: checked if we "overswapped"
    assert np.all(new_data[original_data != label_a] != label_b)
    assert np.all(new_data[original_data != label_b] != label_a)


def test_swaplabel_is_noop_on_identical_labels(segmentation):
    label_a = 1
    label_b = 1
    original_data = np.copy(segmentation.data)
    segmentation.swaplabel(label_a, label_b)
    result_data = segmentation.data
    assert np.array_equal(original_data, result_data), 'Swaplabel should be a no-op here'


def test_fit_from_template_fullchange(segmentation, fullchange_template):
    original_data = np.copy(segmentation.data)
    original_repr = str(segmentation)
    segmentation.fit_to_template(fullchange_template)
    result_data = segmentation.data
    result_repr = str(segmentation)
    
    print(f'Unique original data: {np.unique(original_data)}')
    print('Original: ', original_repr)
    print('Result: ', result_repr)

    assert set(np.unique(result_data)) == set((0, 10, 20, 30)), 'Numerical data should be modified'  











