import numpy as np
import scipy.ndimage as ndimage

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


def rescale(array: np.ndarray, original_voxel_size: float, rescaled_voxel_size: float,
            order: int = 3, mode: str = 'reflect') -> np.ndarray:
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

    order : int, optional   
        The order of the interpolation splines. Defaults to 3

    mode : str, optional
        Deetermines the how the input array is extended beyond
        boundaries. See 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
        for in-depth info.
        Defaults to 'reflect'.
    
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


def transform_ijk_position(ijk_position: Sequence[int],
                           transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Transfrom a 3D IJK position with the 4D transformation matrix in homogenous coordinates.
    Can be utilized to compute landmark IJK position after rescaling of voxel raw data.
    """
    if not isinstance(ijk_position, np.ndarray):
        ijk_position = np.array(ijk_position)
    ijk_transformed_hom = transformation_matrix @ expand_to_4D(ijk_position)
    return reduce_from_4D(ijk_transformed_hom)