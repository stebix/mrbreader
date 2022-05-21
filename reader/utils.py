import numpy as np
import collections
import collections.abc

from typing import List, Dict, Tuple, Union, Iterable, Any


"""
Several utility functions and definitions
"""

int_like_dtypes = (np.int8, np.int16, np.int32, np.int64,
                   np.uint8, np.uint16, np.uint32, np.uint64)

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
        return False

    
def extent_string_as_points(extent: str) -> Tuple[np.ndarray]:
    """
    Interpret an extent string consisting of six integer coordinates,
    like '12 50 125 539 26 121' as a 2-tuple of 3D points:
    start -> [12, 125, 26]    stop -> [50, 539, 121]
    """
    # Clumsy, but explicit unpacking to enforce existence of six coordinates
    (ax0_start, ax0_stop,
     ax1_start, ax1_stop,
     ax2_start, ax2_stop) = (int(c) for c in extent.split(' '))
    start = np.array([ax0_start, ax1_start, ax2_start])
    stop = np.array([ax0_stop, ax1_stop, ax2_stop])
    return (start, stop)


def extent_string_as_array(extent: str) -> np.ndarray:
    """
    interrpret extent string and construct single six-element array
    from it.
    """
    array = np.array([int(c) for c in extent.split(' ')])
    if array.size != 6:
        raise RuntimeError(f'could not find six integers in extent string "{extent}"')
    return array


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
    # enforce integer array argument for normal procedure
    if array.dtype not in int_like_dtypes:
        msg = f'Non-integer label array! Dtype: {array.dtype}'
        raise ValueError(msg)

    labelvalues = np.unique(array)
    expanded_shape = (len(labelvalues),) + array.shape
    onehot_array = np.zeros(expanded_shape, dtype=np.int)
    for idx, labelval in enumerate(labelvalues):
        onehot_array[idx, array == labelval] = 1
    return onehot_array



def relabel(label_array: np.ndarray,
            old: Union[int, Iterable[int]],
            new: Union[int, Iterable[int]]) -> np.ndarray:
    """
    Re-label an array by replacing all occurrences of the old
    label with the new label.
    This is a pure function and thus produces a new array.
    """
    if label_array.dtype not in int_like_dtypes:
        msg = f'Non-integer label array! Dtype: {label_array.dtype}'
        raise ValueError(msg) 

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


def lps_to_ijk_matrix(space_directions: np.ndarray,
                      space_origin: np.ndarray) -> np.ndarray:
    """
    Construct LPS (left posterior superior) image space to voxel space transformation matrix
    `lps_to_ijk` from 3DSlicer/DICOM metadata attributes.

    See: https://www.slicer.org/wiki/Coordinate_systems and
         https://discourse.slicer.org/t/building-the-ijk-to-ras-transform-from-a-nrrd-file/1513

    
    Parameters
    ----------

    space_directions: np.ndarray
        Axis directions as a 3 x 3 matrix/array.

    space_origin: np.ndarray
        Coordinate system origin. Acts as a translation vector.
        3D vector.

    
    Returns
    -------

    transformation_matrix: np.ndarray
        The 4D transformation matrix from LPS to IJK
        in homogenous coordinates.
    """
    assert space_directions.shape == (3, 3), (f'Expecting shape (3, 3) for direction matrix, '
                                              f'got {space_directions.shape}')
    assert space_origin.shape == (3,), (f'Expecting shape (3,) for origin vector, '
                                        f'got {space_origin.shape}')
    unitvec = np.array([[0, 0, 0, 1]])
    space_concat = np.concatenate((space_directions, space_origin.reshape(3, 1)), axis=1)
    homog_concat = np.concatenate((space_concat, unitvec), axis=0)
    return np.linalg.inv(homog_concat)



def ras_to_ijk_matrix(space_directions: np.ndarray,
                      space_origin: np.ndarray) -> np.ndarray:
    """
    Construct RAS (right anterior superior) image space to voxel space transformation matrix
    `ras_to_ijk` from 3DSlicer/DICOM metadata attributes.

    See: https://www.slicer.org/wiki/Coordinate_systems and
         https://discourse.slicer.org/t/building-the-ijk-to-ras-transform-from-a-nrrd-file/1513

    
    Parameters
    ----------

    space_directions: np.ndarray
        Axis directions as a 3 x 3 matrix/array.

    space_origin: np.ndarray
        Coordinate system origin. Acts as a translation vector.
        3D vector.

    
    Returns
    -------

    transformation_matrix: np.ndarray
        The 4D transformation matrix from RAS to IJK
        in homogenous coordinates.

    """
    lps_to_ijk = lps_to_ijk_matrix(space_directions, space_origin)
    ras_to_lps = np.diag([-1, -1, 1, 1])
    # composite matrix acts on vector on right side
    return lps_to_ijk @ ras_to_lps


def is_4D_column_vector(vector: np.ndarray) -> bool:
    """
    Checks if vector candidate is 4D column vector of the form
    [[x]
     [y]
     [z]]    with shape (4, 1)
    """
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    if vector.shape == (4, 1):
        return True
    return False


def expand_to_4D(vector: np.ndarray) -> np.ndarray:
    """Expand flat 3D vector to 4D column vector in homogenous coordinates."""
    expanded_vector = np.concatenate((vector, np.ones(1, vector.dtype)), axis=0)
    return expanded_vector.reshape(4, 1)


def reduce_from_4D(vector: np.ndarray) -> np.ndarray:
    """Reduce 4D vector in homogenous coordinates to flat spatial 3D vector"""
    return np.squeeze(vector)[:-1]