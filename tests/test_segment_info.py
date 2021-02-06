import copy
import numpy as np
import pytest

from reader.segment_info import SegmentInfo


@pytest.fixture
def segmentation_metadata(mrbfile):
    """
    Get the (first instance) segmentation metadata dict
    from the `mrbfile` exemplary file.
    """
    data = mrbfile._read_members(
        mrbfile.segmentation_members,
        mrbfile.read_nrrd
    )
    # get zeroth element of segmentation members list and use second
    # element of the tuple, since `read_nrrd` returns (data, header)
    metadata = data[0][1]
    assert isinstance(metadata, dict), 'Expecting a dict-like object as metadata'
    return metadata


@pytest.fixture
def seginfo(segmentation_metadata):
    """
    Directly construct the `SegmentationInfo` object
    from the test file.
    """
    seginfo = SegmentInfo.from_header(segmentation_metadata)[0]
    return seginfo


def test_initialization(segmentation_metadata):
    """
    Correct construction of SegmentInfo instance from 'real-world'
    metadata dictionary.
    """
    seginfo = SegmentInfo.from_header(segmentation_metadata)
    assert seginfo[0], 'Object creation failed somehow'


def test_equality_operator(segmentation_metadata):
    seginfo_a = SegmentInfo.from_header(segmentation_metadata)[0]
    seginfo_b = copy.deepcopy(seginfo_a)
    seginfo_b.ID = 'Seven_of_Nine'

    assert seginfo_a == seginfo_a, 'This should be identical'
    assert not (seginfo_a == seginfo_b), 'This should be not equal'
    assert seginfo_b == seginfo_b, 'This should be identical'


def test_to_dict_internal(seginfo):
    result = seginfo.to_dict(keystyle='internal')
    for key, value in result.items():
        assert getattr(seginfo, key) == value, 'Dict value should be equal to attr'


def test_color_attr_is_rgb_tuple(seginfo):
    c_length = len(seginfo.color)
    assert c_length == 3, f'Should be an RGB 3-tuple, got length {c_length}'
    for elem in seginfo.color:
        assert elem <= 1.0, f'Should be leq 1, got {elem}'
        assert elem >= 0.0, f'Should be geq 0, got {elem}'


def test_color_setter_valid_string(seginfo):
    new_color = '0.345 0.345 0.234'
    new_color_float = (0.345, 0.345, 0.234)
    seginfo.color = new_color
    for val, expected in zip(seginfo.color, new_color_float):
        assert np.isclose(val, expected), 'Should be equal within machin precision'


@pytest.mark.parametrize('inv_color', ['0.234 0.234253', '1 2 3', '123'])
def test_color_setter_invalid_string_arg(seginfo, inv_color):
    with pytest.raises(AssertionError):
        seginfo.color = inv_color


@pytest.mark.parametrize('inv_color', [(0.234, 0.234253), (1, 2, 3), 123])
def test_color_setter_invaid_numerical_arg(seginfo, inv_color):
    # small explanation of expected failure modes:
    # AssertionError: 1 due to wrong length
    #                 2 due to wrong interval
    # TypeError:  due to len(<class 'int'>)  in assert statement
    with pytest.raises((AssertionError, TypeError)):
        seginfo.color = inv_color


@pytest.mark.parametrize('labelval', [1, 0, 0.25])
def test_label_value_attr(seginfo, labelval):
    seginfo.label_value = labelval
    assert isinstance(seginfo.label_value, int), 'Type should be force-cast to int'
    assert seginfo.label_value == int(labelval), 'This should have been set'


def test_extent_is_valid_index_tuple(seginfo):
    assert len(seginfo.extent) == 6, 'Expecting a 6-tuple'
    for idx in seginfo.extent:
        assert isinstance(idx, int), f'Expecting integer idx, got: {type(idx)}'


def test_extent_setter_valid_string(seginfo):
    valid_extent = '256 512 129 64 32 16'
    valid_extent_int = (256, 512, 129, 64, 32, 16)
    seginfo.extent = valid_extent
    assert seginfo.extent == valid_extent_int, 'Should have been assigned'


def test_extent_setter_valid_int_tuple(seginfo):
    valid_extent_int = (256, 512, 129, 64, 32, 16)
    seginfo.extent = valid_extent_int
    assert seginfo.extent == valid_extent_int, 'Should have been assigned'


@pytest.mark.parametrize('inv_str', ['a b c d e f', '123 14 34 235 9', 'asdfasf'])
def test_extent_setter_invalid_string(seginfo, inv_str):
    with pytest.raises((AssertionError, ValueError)):
        seginfo.extent = inv_str


@pytest.mark.parametrize('inv_num', [(1, 2, 3, 4, 5), 1234])
def test_extent_setter_invalid_numerical(seginfo, inv_num):
    with pytest.raises((AssertionError, TypeError)):
        seginfo.extent = inv_num