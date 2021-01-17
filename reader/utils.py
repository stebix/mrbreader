import numpy as np
import collections

from typing import List, Dict, Tuple, Union, Iterable, Any


"""
Several utility functions and definitions
"""

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
    if not isinstance(old, Iterable):
        old = [old]
    if not isinstance(new, Iterable):
        new = [new]

    # sanity checking
    assert len(old) == len(new), f'Every old label needs a new replacement!'
    int_like = (np.int8, np.int16, np.int32, np.int64,
                np.uint8, np.uint16, np.uint32, np.uint64)
    assert label_array.dtype in int_like, f'Non-integer label array! Dtype: {label_array.dtype}'

    relabeled_array = np.zeros_like(label_array)
    relabeled_array = np.copy(label_array)
    for o, n in zip(old, new):
        relabeled_array[label_array == o] = n
    
    return relabeled_array