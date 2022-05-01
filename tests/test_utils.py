import numpy as np
import pytest

"""
Unittests and integrationtest for the utils module.
"""

from reader.utils import (is_binary, is_onehot,
                          convert_to_intlabel,
                          convert_to_onehot,
                          relabel,
                          lps_to_ijk_matrix,
                          ras_to_ijk_matrix,
                          expand_to_4D,
                          reduce_from_4D,
                          extent_string_as_points)


@pytest.fixture
def onehot_array():
    """
    Provides a 3D (spatial) one-hot array (C x D x H x W)
    with three channels (or classes).
    """
    n_c = 3
    shape = (10, 10, 10)
    array = np.random.default_rng().normal(size=(n_c,) + shape)
    argmax_channel_idcs = np.argmax(array, axis=0)
    onehot_array = np.zeros(array.shape, dtype=np.int)
    for idx in range(n_c):
        onehot_array[idx, argmax_channel_idcs == idx] = 1
    return onehot_array


@pytest.fixture
def intlabel_array():
    """
    Provides stochastically produced 3D (spatial) intlabel array (D x H x W)
    with three classes, i.e.
    (intlabel_array)_ijk € {0, 1, 2}
    """
    n_c = 3
    shape = (10, 10, 10)
    intlabel_array = np.random.default_rng().integers(0, high=n_c,
                                                      size=shape)
    return intlabel_array


@pytest.fixture
def label_array():
    """
    Provides deterministically produced 3D (spatial) intlabel array (D x H x W)
    with three classes and a tiled structure along the first axis, i.e.
    (intlabel_array)_ijk € {1, 2, 3}

    """
    n_c = 3
    shape = (10, 10, 10)
    test_array = np.tile(
        np.array([[1,2,3], [2,1,3], [3, 1, 2]], dtype=np.int),
        (3, 1, 1)
    )
    return test_array


@pytest.mark.parametrize('dtype', [np.int, np.float32, np.int64])
def test_is_binary(dtype):
    testshape = (10, 10, 10)
    all_ones = np.ones(shape=testshape, dtype=dtype)
    all_zeros = np.zeros(shape=testshape, dtype=dtype)
    # test for homogenous base cases
    assert is_binary(all_ones)
    assert is_binary(all_zeros)
    # test mixture
    assert is_binary(np.stack([all_ones, all_zeros], axis=0))
    # random array: may be binary by chance, but this is unlikely ;)
    randarr = np.random.default_rng().normal(size=testshape)
    assert not is_binary(randarr)




class Test_extent_string_as_points:

    def test_valid_input(self):
        input_string = '12 50 125 539 26 121'
        expected_start = np.array([12, 125, 26])
        expected_stop = np.array([50, 539, 121])
        result_start, result_stop = extent_string_as_points(input_string)
        assert np.array_equal(result_start, expected_start)
        assert np.array_equal(result_stop, expected_stop)
    

    def test_malformed_input_missing_coordinate(self):
        # missing one coordinate
        malformed_string = '12 50 125 539 123'
        with pytest.raises(ValueError):
            _ = extent_string_as_points(malformed_string)


    def test_malformed_input_not_castable(self):
        # non integer castable value
        malformed_string = '12 50 125 539 123 400a'
        with pytest.raises(ValueError):
            _ = extent_string_as_points(malformed_string)



class Test_convert_to_intlabel:

    def test_fail_on_nonbinary(self):
        shape = (10, 10, 10)
        arr = np.random.default_rng().normal(size=shape)
        with pytest.raises(AssertionError):
            convert_to_intlabel(arr)

    
    def test_correct_execution(self, onehot_array):
        """
        Tests the correct execution of conversion to intlabel array
        on 3D onehot array.
        """
        # actually use function to produce an intlabel array
        intlabel_array = convert_to_intlabel(onehot_array)
        assert intlabel_array.shape == onehot_array.shape[1:], 'Spatial shapes should match'

        # check that correct int labels exist at the positions indicated
        # in the on-hot encoded arrays
        for idx in range(onehot_array.shape[0]):
            channel_arr = onehot_array[idx, ...].astype(np.bool)
            assert is_binary(channel_arr)
            assert np.allclose(intlabel_array[channel_arr], idx)


    def test_is_inverse_of_convert_to_onehot(self, onehot_array):
        """
        Test the bidrectional nature of the conversion relation
        to_onehot <-> to_intlabel 
        """
        intlabel_array = convert_to_intlabel(onehot_array)
        converted_onehot_array = convert_to_onehot(intlabel_array)

        assert np.allclose(converted_onehot_array, onehot_array)



class Test_convert_to_onehot:

    def test_fail_on_nonint_array(self):
        test_array = np.random.default_rng().normal(size=(10, 10, 10))
        with pytest.raises(ValueError):
            convert_to_onehot(test_array)
    
    def test_correct_execution(self, intlabel_array):
        """
        Test the correct execution of conversion to one-hot encoded
        array from a 3D intlabel array.
        """
        onehot_array = convert_to_onehot(intlabel_array)
        for idx in np.unique(intlabel_array):
            manual_onehot = intlabel_array == idx
            assert np.array_equal(onehot_array[idx, ...], manual_onehot)

    
    def test_is_inverse_of_convert_to_intlabel(self, intlabel_array):
        """
        Test again the bideirectional nature of the conversion relation
        to_intlabel <-> to_onehot
        """
        onehot_array = convert_to_onehot(intlabel_array)
        computed_intlabel_aray = convert_to_intlabel(onehot_array)

        assert np.allclose(computed_intlabel_aray, intlabel_array) 



class Test_relabel:

    @pytest.mark.parametrize('dtype', [np.float32, np.complex])
    def test_fail_on_nonint_array(self, dtype):
        """
        Routine is expected to blow up for non-integer arrays.
        """
        shape = (5, 5, 5)
        test_array = np.random.default_rng().normal(size=shape)
        with pytest.raises(ValueError):
            res = relabel(test_array.astype(dtype), 0, 1)

    
    def test_single_integer_argument(self, label_array):
        """
        Function accepts both single ints and lists of ints for
        relabeling elements.
        Here we test the single integer case. 
        """
        # value swaps
        old = 0
        new = 1
        mask = label_array == old
        res = relabel(label_array, old, new)
        # check correct relabeling of old - new pair
        assert np.allclose(label_array[mask], res[mask]), 'Should change from old to new!'
        # check that other values are not modified
        assert np.allclose(label_array[~mask], res[~mask]), 'Should not change!'

    
    def test_list_integer_argument(self, label_array):
        """
        Test the function for a list of integers.
        """
        old = [1, 2, 3]
        new = [100, 200, 300]
        masks = [label_array == o for o in old]
        res = relabel(label_array, old, new)
        for mask, n in zip(masks, new):
            assert np.allclose(res[mask], n), 'Value should be relabeled!'
        

    def test_is_noop_if_old_not_present(self, label_array):
        old = 42
        new = 0
        res = relabel(label_array, old, new)
        assert np.allclose(res, label_array), 'Array should remain unmodified!'
        

    def test_relabel_to_homogenous(self, label_array):
        """
        Test that a full relabeling of multiple old values to
        a single new value produces a homogenous array.
        """
        old = np.unique(label_array)
        new = np.repeat([-137], old.shape)
        res = relabel(label_array, old, new)
        assert np.allclose(res, new), 'Should be filled with a single value'


class Test_coordinate_transformations:


    def test_effect_lps_to_ijk(self):
        expected_inverse = np.array(
            [[1, 2, 3, 8],
             [2, 1, -1, 6],
             [4, 9, 0, 1],
             [0, 0, 0, 1]]
        )
        space_direction = expected_inverse[:3, :3]
        space_origin = expected_inverse[:-1, -1]
        expected_result = np.linalg.inv(expected_inverse)

        result = lps_to_ijk_matrix(space_directions=space_direction,
                                   space_origin=space_origin)

        assert np.allclose(result, expected_result), 'Result transformation matrix mismatch'
        

def test_expand_4D():
    vector = np.array([0, 5, 9])
    expected_result = np.array([0, 5, 9, 1]).reshape(4, 1)
    result = expand_to_4D(vector)
    assert expected_result.shape == result.shape, (f'Shape mismatch: {result.shape} '
                                                   f'!= {expected_result.shape} (expected)')
    assert np.allclose(expected_result, result), 'Result mismatch'


def test_reduce_4D():
    vector = np.array([0, 5, 9, 1]).reshape(4, 1)
    expected_result = np.array([0, 5, 9])
    result = reduce_from_4D(vector)
    assert expected_result.shape == result.shape, (f'Shape mismatch: {result.shape} '
                                                   f'!= {expected_result.shape} (expected)')
    assert np.allclose(expected_result, result), 'Result mismatch'

