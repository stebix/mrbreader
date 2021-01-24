import numpy as np
import pytest

"""
Unittests and integrationtest for the utils module.
"""

from reader.utils import is_binary, convert_to_intlabel


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


class Test_convert_to_intlabel:

    def test_fail_on_nonbinary(self):
        shape = (10, 10, 10)
        arr = np.random.default_rng().normal(size=shape)
        with pytest.raises(AssertionError):
            convert_to_intlabel(arr)

    
    def test_correct_execution(self):
        """
        Correct execution and labeling on 3D array.
        """
        n_c = (3,)
        shape = (10, 10, 10)
        array = np.random.default_rng().normal(size=n_c + shape)
        onehot_array = np.zeros_like(array)
        onehot_array[np.argmax(array, axis=0), ...] = 1

        assert is_onehot(onehot_array)