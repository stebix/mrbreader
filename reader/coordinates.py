import numpy as np
import numba as nb

from copy import copy
from itertools import product
from typing import Dict, List, Sequence, Tuple, Optional


# def ijk_to_lps_matrix(space_directions, space_origin):
#     unitvec = np.array([[0, 0, 0, 1]])
#     space_concat = np.concatenate((space_directions, space_origin.reshape(3, 1)), axis=1)
#     homog_concat = np.concatenate((space_concat, unitvec), axis=0)
#     return homog_concat

# def ijk_to_ras_matrix(space_directions, space_origin):
#     unitvec = np.array([[0, 0, 0, 1]])
#     space_concat = np.concatenate((space_directions, space_origin.reshape(3, 1)), axis=1)
#     homog_concat = np.concatenate((space_concat, unitvec), axis=0)
#     return homog_concat



def ijk_coordinate_array(shape: Sequence[int]) -> np.ndarray:
    """
    Get the IJK coordinate vectors of the arrays with `shape` in homogenous
    4D coordinates.

    Parameters
    ----------

    shape : Sequence of int
        The shape of the base array for which the IJK coordinate
        vectors are to be obtained.

    Returns
    -------

    homogenous_ijk_coordinates: np.ndarray
        The 4D IJK coordinate vectors. 
    """
    ijk_coordinates = np.meshgrid(
        *[np.arange(axis_size) for axis_size in shape],
        indexing='ij'
    )
    homogenous_expansion = [*ijk_coordinates, np.ones_like(ijk_coordinates[0])]
    return np.stack(homogenous_expansion, axis=-1)


@nb.jit(nopython=True, fastmath=True)
def _transform_vectors(vectors: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform array of vectors with transformation T.
    This is a numba high-performance, JIP-compiled implementation.

    Parameters
    ----------

    vectors : np.ndarray
        The array of vectors on which the linear transformation matrix
        will be applied.
        Must be a 2D-array of shape (M, vectordim) where `vectordim`
        is the dimensionality of the vectors.

    T : np.ndarray
        The transformation matrix. Must be of shape (N, vectordim)

    Returns
    -------

    transformed_vectors: np.ndarray
        The array of length M in first dimension containing the
        transformed vectors.
    """
    transformed_vectors = np.zeros_like(vectors)
    for i in range(vectors.shape[0]):
        transformed_vectors[i] = T @ vectors[i]
    return transformed_vectors


def xyz_coordinate_array(shape: Sequence[int],
                         T_ijk_to_xyz: np.ndarray,
                         flatten: bool = True) -> np.ndarray:
    """
    Get the XYZ coordinates of the regular grid array with the given shape.

    Parameters
    ----------

    shape : Sequence of int
        The shape of the data array (assumed to be regularly gridded).

    T_ijk_to_xyz : np.ndarray
        The (4 x 4) transformation matrix in homogenous coordinates
        that transforms from IJK voxel coordinates to XYZ physical
        coordinates.
    
    flatten : bool, optional
        Return a flattened array of shape (I_x * I_y * I_z, 4) containing the 
        XYZ coordinate vectors. Defaults to *True*

    Returns
    -------

    xyz_coordinates : np.ndarray
        The physical coordinate vector array in homogenous, 4D
        cooridnates. 
    """
    # reshape and recast prior to numba deployment
    coordinate_vectors = ijk_coordinate_array(shape)
    input_shape = coordinate_vectors.shape
    coordinate_vectors = np.reshape(coordinate_vectors,
                                    newshape=(-1, input_shape[-1]))
    coordinate_vectors = coordinate_vectors.astype(np.float32)
    T_ijk_to_xyz = T_ijk_to_xyz.astype(np.float32)
    xyz_coordinates = _transform_vectors(coordinate_vectors, T_ijk_to_xyz)
    if flatten:
        return xyz_coordinates
    else:
        return np.reshape(xyz_coordinates, newshape=(*input_shape, 4))




def ijk_to_xyz_matrix(space_directions: np.ndarray, space_origin: np.ndarray) -> np.ndarray:
    """
    Generate the transformation matrix from IJK voxel coordinates to XYZ physical
    coordinates in homogenous, i.e. 4D form. 
    This matrix converts voxel coordinates to physical coordinates. Warning:
    At this stage it is generally undefined whether XYZ result is in RAS or LPS system!

    Parameters
    ----------

    space_directions : np.ndarray
        The (3 x 3) space directions matrix.
    
    space_origin : np.ndarray
        The (3,) space origin vector.
    
    Returns
    -------

    T : np.ndarray
        the transformation matrix IJK -> XYZ in homogenous cooridnates.
    """
    unitvec = np.array([[0, 0, 0, 1]])
    space_concat = np.concatenate((space_directions, space_origin.reshape(3, 1)), axis=1)
    homog_concat = np.concatenate((space_concat, unitvec), axis=0)
    return homog_concat


def clean_space_directions(space_directions: np.ndarray) -> np.ndarray:
    """Clean possible NaN-tainted space directions array"""
    if not np.any(np.isnan(space_directions)):
        return space_directions
    # first row of (4 x 3) matrix NaN?
    if np.all(space_directions[0, :]) and space_directions.shape == (4, 3):
        return space_directions[1:, :]

    raise ValueError('Strict cleaning conditions not met: Invalid space directions matrix!')


def ijk_to_xyz_matrix_from_metadata(metadata: Dict, target_coordsystem: str) -> np.ndarray:
    """
    Directly generate the IJK -> XYZ transformation matrix from a metadata
    dictionary produced by 3DSlicer, ITKsnap, etc. ...

    Parameters
    ----------

    metadata : dictionary
        The metadata dictionary, here assumed to be generated by 3DSlicer.
        Must hold key information about coordinate system and physical space
        information.

    target_coordsystem : str
        Target coordinate system. Depending on the base coordinate system specified
        by the metadata. A RAS -> LPS or LPS -> RAS transformation
        might be integrated into the transformation matrix result.
        Must be 'lps' or 'ras'.

    Returns
    -------

    T : np.ndarray
        The transformation matrix that maps from IJK system to
        the specified target coordinate system, i.e. RAS or LPS
        XYZ physical system.
    """
    # we need this info: get custom messages instead of anonymous KeyError
    assert 'space directions' in metadata, 'key missing: space direction information'
    assert 'space origin' in metadata, 'key missing: space origin information'
    assert 'space' in metadata, 'key missing: coordinate space information'
    # apply cleaning: Sometimes a wild NaN row appears!?
    spac_dir = clean_space_directions(metadata['space directions'])
    spac_ori = metadata['space origin']

    target_coordsystem = target_coordsystem.lower()
    assert target_coordsystem in ('lps', 'ras'), f'Coordinate system must be RAS or LPS. Got: {coordsystem}'
    base_coordsystem = 'lps' if metadata['space'] == 'left-posterior-superior' else 'ras'

    if base_coordsystem == target_coordsystem:
        switch_ras_lps = np.eye(4)
    else:
        switch_ras_lps = np.diag([-1, -1, 1, 1])

    return switch_ras_lps @ ijk_to_xyz_matrix(spac_dir, spac_ori)


def compute_ijk_edge_coords(array_shape: Sequence[int]) -> List[np.ndarray]:
    """
    Get edge coordinates of 3D cuboids as arrays of shape (3,) from the shape tuple.

    We assume that the array_shape corresponds to an array specifying a 3D cube, like:
      _____
     /     /|
    /_____/ |
    |     | /
    |_____|/  <- it has 8 edge point coordinates { (x_i, y_i, z_i) }

    This function return the edge coordinate vectors in IJK space in homogenous
    coordinates (4D)

    Parameters
    ----------

    array_shape : Sequence of int
        The shape of the array.

    Returns
    -------

    edge_coords : List of np.ndarray
        The list of the 8 edge coordinate vectors
        in homogenous 4D IJK space.
    """
    edge_coords = [
        np.array([i, j, k, 1])
        for (i,j,k) in product(*((0, size-1) for size in array_shape))
    ]
    return edge_coords



def compute_axis_slices(edge_coordinate_array: List[np.ndarray]) -> Tuple[slice]:
    """
    Compute axis slices from an edge coordinate array.
    Expected layout of edge cooridnate array is a list of vectors (np.ndarrays).
    """
    edge_coordinate_array = np.stack(edge_coordinate_array, axis=0)
    axis_slices = []
    for axis in range(3):
        axis_slices.append(slice(edge_coordinate_array[:, axis].min(), edge_coordinate_array[:, axis].max()))
    
    return tuple(axis_slices)


def as_int_slices(slices: Sequence[slice], reterrs: bool = False) -> Tuple[slice]:
    """
    Recast a sequence of slice objects such that the (start, stop) attribute is
    safely transformed to integers via `np.rint` and `int`.
    Computes the start - stop difference error if `reterrs` flag is set.
    """
    if isinstance(slices, slice):
        slices = [slices]
    integer_slices = []
    errors = []
    for s in slices:
        int_slice = slice(
            max((0, int(np.rint(s.start)))),
            int(np.rint(s.stop)) + 1
        )
        diff_float = (s.stop + 1) - max(0, s.start)
        diff_int = int_slice.stop - int_slice.start
        errors.append(abs(diff_float - diff_int))
        integer_slices.append(int_slice)
    if reterrs:
        return (tuple(integer_slices), tuple(errors))
    else:
        return tuple(integer_slices)


def transform_vectors(vectors: Sequence[np.ndarray], T_matrix: np.ndarray) -> List[np.ndarray]:
    """
    Compute the linear transformation of the sequence of vectors by applying
    the map specified through the transformation matrix to every vector in vectors.

    Parameters
    ----------

    vectors : Sequence of np.ndarray
        The sequence vectors which are transformed by T_matrix

    T_matrix : np.ndarray
        The transformation matrix. 
    """
    return [T_matrix @ v for v in vectors]



def compute_xyz_edge_coords(ijk_edge_coords: np.ndarray, T_matrix: np.ndarray) -> List[np.ndarray]:
    """
    Compute edge coordinates in physical XYZ space from IJK vectors
    via the map specified through the transformation matrix

    Parameters
    ----------

    ijk_edge_coords : np.ndarray
        The IJK coordinate vectors 
    
    """
    return [T_matrix @ ijk_vector for ijk_vector in ijk_edge_coords]



def recompute_ijk_positions(raw_crop_slices: Sequence[slice], landmarks: Dict) -> Dict:
    """
    Recompute IJK positions of landmarks from a cropping 3D slice specification.
    """
    recomputed_landmarks = []
    for landmark in landmarks:
        adjusted_landmark = copy(landmark)
        adjusted_landmark['ijk_position'] = tuple(
            (elem - s.start for elem, s in zip(landmark['ijk_position'], raw_crop_slices))
        )
        recomputed_landmarks.append(adjusted_landmark)
    return recomputed_landmarks

