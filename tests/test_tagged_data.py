import copy
import pathlib
import numpy as np
import pytest

from reader.mrbfile import MRBFile
from reader.tagged_data import SegmentationData
from reader.segment_info import SegmentInfo


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
    """
    Test that all SegmentInfo attributes and the corresponding data numpy.ndarray are changed
    by the fit_from_template method.
    """
    original_data = np.copy(segmentation.data)
    original_repr = str(segmentation)
    segmentation.fit_to_template(fullchange_template)
    result_data = segmentation.data
    result_repr = str(segmentation)
    
    print(f'Unique original data: {np.unique(original_data)}')
    print('Original: ', original_repr)
    print('Result: ', result_repr)

    # check correct fit of numerical data
    old_labels = np.unique(original_data)
    new_labels = np.unique(result_data)

    for o, n in zip(old_labels, new_labels):
        assert np.allclose(result_data[original_data == o], n), f'Expected label change {o} -> {n}!'




class Test_fit_from_template_multiple_segmentations:
    """
    This functionality is quite complex to test. Thus, various test scenarios are split into
    multiple test functions packed into a single test class.
    """
    
    @pytest.fixture
    def segmentation_1(self, mock_label_data):
        """
        Real test file #1 
        """
        fpath = pathlib.Path(
            'C:/Users/Jannik/Desktop/mrbreader/tests/assets/testmrb_multiseg.mrb')
        mrb = MRBFile(fpath)
        seg = mrb.read_segmentations()[0]
        # insert smaller mock label data to ensure good testing performance
        seg.data = mock_label_data
        return seg

    @pytest.fixture
    def segmentation_2(self, mock_label_data):
        """
        Real test file #2 
        """
        fpath = pathlib.Path(
            'C:/Users/Jannik/Desktop/mrbreader/tests/assets/testmrb_multiseg_2.mrb')
        mrb = MRBFile(fpath)
        seg = mrb.read_segmentations()[0]
        # insert smaller mock label data to ensure good testing performance
        seg.data = mock_label_data
        return seg


    @pytest.fixture
    def segmentation_3(self, mock_label_data):
        """
        Real test file #3 
        """
        fpath = pathlib.Path(
            'C:/Users/Jannik/Desktop/mrbreader/tests/assets/testmrb_multiseg_3.mrb')
        mrb = MRBFile(fpath)
        seg = mrb.read_segmentations()[0]
        # insert smaller mock label data to ensure good testing performance
        seg.data = mock_label_data
        return seg


    @pytest.fixture
    def segmentations(self, segmentation_1, segmentation_2, segmentation_3):
        """
        List collection of the three real es files.
        The naming and labeling schemes differ fo testing purposes
        """
        return [segmentation_1, segmentation_2, segmentation_3]


    def test_print_string_reprs(self, segmentations, fullchange_template):
        """
        Homogenize the different segmentations and print the object string
        representations to manually inspect them.
        This works only if `stdout` is not captured by pytest.
        """
        for segment in segmentations:
            segment.fit_to_template(fullchange_template)
            print(segment)

    
    def test_data_members_all_relabeled_correctly(self, segmentations, fullchange_template):
        """
        Test that all data members are relabeled correctly. Four our test setup this checks
        roughly that the following transformation of the label volumes is applied:

        0 1 3      0  10 30
        2 1 0  ->  20 10 0
        0 0 0      0  0  0
        """
        # original segmentations
        og_segmentations = []

        # apply transformation to adhere to template
        for seg in segmentations:
            og_segmentations.append(copy.deepcopy(seg))
            seg.fit_to_template(fullchange_template)

        # check that background is preserved correctly
        for ogseg, seg in zip(og_segmentations, segmentations):
            assert np.all(~seg.data[ogseg.data == 0]), 'Background should be unmodified'
            assert np.all(seg.data[ogseg.data != 0].astype(np.bool)), 'Expecting no spillover from background to other'

        # check the other label values
        for og_seg, seg in zip(og_segmentations, segmentations):
            # find out the label value pairs (old, new) for the
            # segments in the current segmentation object
            old_data = og_seg.data
            new_data = seg.data

            for og_si in og_seg.infos.values():
                for equiv_set, target_spec in fullchange_template.items():
                    if og_si.name in equiv_set:
                        # label value pair
                        old_lv, new_lv = (og_si.label_value, target_spec['label_value'])
                        assert np.allclose(new_data[old_data == old_lv], new_lv), 'Label value should be changed'

