import numpy as np
import cupy as cp
import scipy.ndimage as ndimage
import cupyx.scipy.ndimage as cupyndimage

from typing import Sequence

from reader.utils import expand_to_4D, reduce_from_4D


def rescaled_shape(shape: tuple, rescaling_factor: float) -> tuple:
    """Compute new shape after rescaling an array by the given rescaling_factor."""
    return tuple(int(round(size*(1/rescaling_factor))) for size in shape)


def create_rescaling_matrix_2D(original_pixel_size: float, rescaled_pixel_size: float) -> np.ndarray:
    """Compute the 2D transformation matrix in homogenous coordinates that performs a rescaling transformation."""
    rescaling_factor = rescaled_pixel_size / original_pixel_size
    row0 = (rescaling_factor, 0, 0)
    row1 = (0, rescaling_factor, 0)
    row2 = (0, 0, 1)
    return np.asarray([row0, row1, row2], dtype=np.float64)


def create_rescaling_matrix(original_voxel_size: float, rescaled_voxel_size: float) -> np.ndarray:
    """Compute the 3D transformation matrix in homogenous coordinates that performs a rescaling transformation."""
    rescaling_factor = rescaled_voxel_size / original_voxel_size
    row0 = (rescaling_factor, 0, 0, 0)
    row1 = (0, rescaling_factor, 0, 0)
    row2 = (0, 0, rescaling_factor, 0)
    row3 = (0, 0, 0, 1)
    return np.asarray([row0, row1, row2, row3], dtype=np.float64)


def rescale(array: np.ndarray,
            original_voxel_size: float, rescaled_voxel_size: float,
            order: int, mode: str) -> np.ndarray:
    """
    Rescale an array using the original and rescaled voxel sizes.

    Parameters
    ----------

    array : np.ndarray
        The input array

    original_voxel_size : float
        The original voxel size (isotropic).
    
    rescaled_voxel_size : float
        The target rescaled voxel size (isotropic).

    order : int   
        The order of the interpolation splines.

    mode : str
        Determines the how the input array is extended beyond
        boundaries. See 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
        for in-depth info.
    
    Returns
    -------

    transformed_array : np.ndarray
        The rescaled and thusly transformed array.
    """
    rescaling_factor = rescaled_voxel_size / original_voxel_size
    target_shape = rescaled_shape(array.shape, rescaling_factor)
    transformation_matrix = create_rescaling_matrix(original_voxel_size, rescaled_voxel_size)
    return ndimage.affine_transform(array, transformation_matrix, order=order,
                                    output_shape=target_shape, mode=mode)


def rescale_gpu(array: np.ndarray,
                original_voxel_size: float, rescaled_voxel_size: float,
                order: int, mode: str) -> np.ndarray:
    """
    Performs rescaling operation via cupy on GPU by auto-transfer of array
    from and to host after computation.
    """
    result = _rescale_gpu(cp.array(array), original_voxel_size,
                          rescaled_voxel_size, order, mode)
    return cp.asnumpy(result)


def _rescale_gpu(array: cp.ndarray,
                 original_voxel_size: float, rescaled_voxel_size: float,
                 order: int, mode: str) -> cp.ndarray:
    rescaling_factor = rescaled_voxel_size / original_voxel_size
    target_shape = rescaled_shape(array.shape, rescaling_factor)
    transformation_matrix = cp.array(
        create_rescaling_matrix(original_voxel_size, rescaled_voxel_size)
    )
    return cupyndimage.affine_transform(array, transformation_matrix, order=order,
                                        output_shape=target_shape, mode=mode)



def transform_ijk_position(ijk_position: Sequence[int],
                           matrix: np.ndarray,
                           intcast: bool = True) -> np.ndarray:
    """
    Transform a 3D IJK position with the 4D transformation matrix in homogenous coordinates.
    Can be utilized to compute landmark IJK position after rescaling of voxel raw data.

    Parameters
    ----------

    ijk_position : Sequence of int
        IJK voxel coordinate position in original scaled grid.

    matrix : np.ndarray
        (4, 4) - Shaped transformation matrix in homogenous coordinates.

    intcast: bool, optional
        Switch casting behaviour of resulting scaled IJK coordinates to integer
        (both rounding and actual integer typecasting). Defaults to True. 
    """
    if not isinstance(ijk_position, np.ndarray):
        ijk_position = np.array(ijk_position)
    ijk_transformed = reduce_from_4D(matrix @ expand_to_4D(ijk_position))
    if not intcast:
        return ijk_transformed
    return np.rint(ijk_transformed).astype(np.int64)


def create_space_direction_matrix(voxel_size: float, system: str) -> np.ndarray:
    """
    Create a (3 x 3) space direction matrix from a isotropic voxel size specification
    and a coordinate system specification (LPS or RAS)
    """
    if system == 'LPS':
        return np.diag([-voxel_size, -voxel_size, +voxel_size])
    elif system == 'RAS':
        return np.diag([voxel_size, voxel_size, voxel_size])
    else:
        valid_systems = ('LPS', 'RAS')
        raise ValueError(
            f'Invalid coordinate system. Must be in {valid_systems} but got {system}'
        )
