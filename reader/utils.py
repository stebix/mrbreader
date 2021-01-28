import numpy as np
import collections
import collections.abc

from typing import List, Dict, Tuple, Union, Iterable, Any


"""
Several utility functions and definitions
"""

def is_binary(array: np.ndarray) -> bool:
    """
    Check if given array is binary. Performs a 'soft check',
    i.e. if all elements are clos to either 1 or 0.
    """
    if array.dtype == np.bool:
        return True
    elif np.all(np.logical_or(np.isclose(array, 1), np.isclose(array, 0))):
        return True
    else:
        return False


def is_onehot(array: np.ndarray, axis: int = 0) -> bool:
    """
    Check if given array is one-hot along a given axis.
    """
    if not is_binary(array):
        return False

    reduced_array = np.sum(array, axis=axis)
    if np.allclose(reduced_array, 1):
        return True
    else:
        print(f'max is {np.max(reduced_array)}')
        return False





def convert_to_intlabel(array: np.ndarray) -> np.ndarray:
    """
    Convert an onehot-encoded label array of the form (C x [... spatial ...])
    to consecutive integer label array.
    Different classes/channels along the first axis are labeled with consecutive
    integers.

    Parameters
    ----------

    array : np.ndarray
        The input array. Expected to be one-hot-encoded and
        of the form (C x [...spatial...])
    
    Returns
    -------

    intlabel_array : np.ndarray
        The converted array with consecutive integer labels.
        has the form ([...spatial...]) due to reduction
        over channel axis.
    """
    assert is_binary(array), 'Expecting a binary-like array'
    # reduce over channel axis
    contracted_shape = array.shape[1:]
    intlabel_array = np.zeros(contracted_shape, dtype=np.int)
    # zero stays zero: background
    for idx in range(1, array.shape[0]):
        mask = array[idx, ...].astype(np.bool)
        intlabel_array[mask] = idx
    return intlabel_array
    

def convert_to_onehot(array: np.ndarray) -> np.ndarray:
    """
    Convert a multi-integer label array to its one-hot encoded
    representation.
    The number of channels/classes are deduced automatically from the
    unique elements in the input array.
    The returned array has the shape (C x *original_shape*).
    The binary masks along the channel dimensions are ordered
    following the label integers.

    Parameters
    ----------

    array : numpy.ndarray
        The multi-integer label array.

    
    Returns
    -------

    onehot_array : numpy.ndarray
        The one-hot-encoded representation
    """
    labelvalues = np.unique(array)
    expanded_shape = (len(labelvalues),) + array.shape
    onehot_array = np.zeros(expanded_shape, dtype=np.int)
    for idx, labelval in enumerate(labelvalues):
        onehot_array[idx, array == labelval] = 1
    return onehot_array



def homogenize(segmentation_data_list: List) -> None:
    """
    Homogenize segment label_value and naming schemes that might differ
    due to varying individual segmentation schemes.
    """
    pass


def relabel(label_array: np.ndarray,
            old: Union[int, Iterable[int]],
            new: Union[int, Iterable[int]]) -> np.ndarray:
    """
    Re-label an array by replacing all occurrences of the old
    label with the new label.
    """
    int_like = (np.int8, np.int16, np.int32, np.int64,
                np.uint8, np.uint16, np.uint32, np.uint64)
    assert label_array.dtype in int_like, f'Non-integer label array! Dtype: {label_array.dtype}'

    as_iterables = []
    for arg in [old, new]:
        if not isinstance(arg, collections.abc.Iterable):
            as_iterables.append([arg])
        else:
            as_iterables.append(tuple(arg))
    
    (old, new) = as_iterables

    # sanity checking
    assert len(old) == len(new), f'Every old label needs a new replacement!'

    # relabeled_array = np.zeros_like(label_array)
    relabeled_array = np.copy(label_array)
    for o, n in zip(old, new):
        relabeled_array[label_array == o] = n
    
    return relabeled_array