import numpy as np

from itertools import chain
from copy import deepcopy
from typing import Sequence, Dict

from reader.rescaletools import (create_space_direction_matrix, rescale,
                                 create_rescaling_matrix, rescale_gpu,
                                 rescaled_shape,
                                 transform_ijk_position)



class RescalingError(Exception):
    pass


class MetadataRescalingError(RescalingError):
    pass


COORDINATE_SYSTEMS_MAPPING = {
    'left-posterior-superior' : 'LPS',
    'right-anterior-superior' : 'RAS'
}



class Rescaler:
    """
    Performs rescaling operations to modify voxel sizes in 3D CT scans that
    are passed trough a 3DSlicer-based pipeline.
    Thusly the class offers methods to 'rescale'
        - volumes (basal task for 3D numpy arrays)
        - landmarks (transform single IJK voxel cooridnates)
        - 3DSlicer header files (transform space directions, volume extents, etc.)

    Parameters
    ----------

    original_voxel_size : float
        Must have same unit as rescaled voxel size.

    rescaled_voxel_size : float
        Must have same unit as rescaled voxel size.

    raw_interpolation_order : int
        Order of the interpolation spline applied to raw data.
    
    label_interpolation_order : int
        Order of the interpolation spline applied to integer label data.


    mode : str, optional
        Determines the how the input array is extended beyond
        boundaries. See 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
        for in-depth info.
        Defaults to 'reflect'.  
    """

    def __init__(self,
                 original_voxel_sizes: Sequence[float],
                 rescaled_voxel_sizes: Sequence[float],
                 raw_interpolation_order: int,
                 label_interpolation_order: int,
                 mode: str = 'reflect',
                 dtype: np.dtype = np.float32,
                 device: str = 'cpu') -> None:

        self._original_voxel_sizes = original_voxel_sizes
        self._rescaled_voxel_sizes = rescaled_voxel_sizes
        self._raw_interpolation_order = raw_interpolation_order
        self._label_interpolation_order = label_interpolation_order
        self._mode = mode
        self.dtype = dtype
        self._rescaling_factors = np.array(rescaled_voxel_sizes) / np.array(original_voxel_sizes)
        self._matrix = create_rescaling_matrix(
            original_voxel_sizes=original_voxel_sizes,
            rescaled_voxel_sizes=rescaled_voxel_sizes
        )
        self._inverse_matrix = np.linalg.inv(self.matrix)
        self._init_device(device)


    def _init_device(self, device: str) -> None:
        if device == 'cpu':
            self._rescale_fn = rescale
        elif device == 'gpu':
            self._rescale_fn = rescale_gpu
        else:
            raise RuntimeError(f'unsupported device "{device}"')
        self._device = device
        
    
    def rescale_volume(self, array: np.ndarray) -> np.ndarray:
        """
        Rescale a 3D volume using the set original and rescaled voxel sizes.
        """
        assert array.ndim == 3, f'expected 3D array, got ndim = {array.ndim}'
        array = array.astype(self.dtype)
        interpolation_order = self._raw_interpolation_order
        result = self._rescale_fn(array, self.original_voxel_sizes, self.rescaled_voxel_sizes,
                                  interpolation_order, self.mode)
        return result
    

    def rescale_label_volume(self, array: np.ndarray) -> np.ndarray:
        """
        Rescale a 3D label volume. This uses a locally set interpolation order to enhance
        interpolation of crisp integer labels. TODO: Improve this. 
        """
        assert array.ndim == 3, f'expected 3D array, got ndim = {array.ndim}'
        array = array.astype(self.dtype)
        interpolation_order = self._label_interpolation_order
        result = self._rescale_fn(array, self.original_voxel_sizes, self.rescaled_voxel_sizes,
                                  interpolation_order, self.mode)
        return result
    

    def rescale_landmarks(self, landmark_dicts: Sequence[Dict]) -> Sequence[Dict]:
        """
        Apply the affine transformation of the rescaling operation to a sequence
        of landmark dicts, effectively transforming their `ijk_position` value
        to the corresponding voxel position in the rescaled volume. 
        """
        # We want to return a copy and not modify the original dictionaries in place.
        rescaled_landmark_dicts = []
        key = 'ijk_position'
        for landmark_dict in landmark_dicts:
            modified_landmark_dict = deepcopy(landmark_dict)
            modified_landmark_dict.update(
                {key : transform_ijk_position(landmark_dict[key], self.inverse_matrix)}
            )
            rescaled_landmark_dicts.append(modified_landmark_dict)
        return rescaled_landmark_dicts

    
    def rescale_general_metadata(self, metadata: dict) -> dict:
        """
        Update the general metadata dictionary of the rescaled volume to
        reflect the modified voxel sizes.
        """
        # Avoid in-place modification of the method argument.
        updated_metadata = deepcopy(metadata)
        # remap coordinate system strings, e.g.  'left-posterior-superior' -> 'LPS'
        system = COORDINATE_SYSTEMS_MAPPING[metadata['space']]
        # compute rescaled values
        tentative_sizes = rescaled_shape(
            metadata['sizes'], self.original_voxel_sizes, self.rescaled_voxel_sizes
        )
        tentative_space_direction = create_space_direction_matrix(self.rescaled_voxel_sizes,
                                                                  system)
        if metadata['dimension'] == 4:
            tentative_space_direction = np.concatenate(
                [np.full(shape=(1, 3), fill_value=np.nan), tentative_space_direction]
            )
        # push updated information into the dictionary
        updated_metadata['space directions'] = tentative_space_direction
        updated_metadata['sizes'] = np.array(tentative_sizes)
        return updated_metadata


    def rescale_segment_metadata(self, metadata: dict) -> dict:
        updated_metadata = deepcopy(metadata)
        # Filter for `SegmentN_Extent` keys and exclude background.
        keys_to_update = set(
            key for key, value in metadata.items() if key.endswith('Extent') and value != 'none'
        )
        for key in keys_to_update:
            start, stop = updated_metadata[key][::2], updated_metadata[key][1::2]
            rescaled_start = self.rescale_ijk_position(start)
            rescaled_stop = self.rescale_ijk_position(stop)
            rescaled_extent = tuple(chain(*zip(rescaled_start, rescaled_stop)))
            updated_metadata[key] = rescaled_extent
        return updated_metadata


    def rescale_segment_infos(self, infos: dict) -> dict:
        updated_infos = deepcopy(infos)
        for labelvalue, segmentinfo in updated_infos.items():
            if labelvalue == 0:
                # labelvalue zero is background and has no extent
                continue
            start, stop = segmentinfo.extent[::2], segmentinfo.extent[1::2]
            rescaled_start = self.rescale_ijk_position(start)
            rescaled_stop = self.rescale_ijk_position(stop)
            rescaled_extent = tuple(chain(*zip(rescaled_start, rescaled_stop)))
            segmentinfo.extent = rescaled_extent
        return updated_infos

    
    def rescale_ijk_position(self, position: np.ndarray) -> np.ndarray:
        """
        Compute the new location of the 3D IJK coordinate point in
        a rescaled volume.
        """
        return transform_ijk_position(position, self.inverse_matrix)


    @property
    def original_voxel_sizes(self) -> float:
        return self._original_voxel_sizes
    
    @property
    def rescaled_voxel_sizes(self) -> float:
        return self._rescaled_voxel_sizes
    
    @property
    def rescaling_factors(self) -> float:
        return self._rescaling_factors

    @property
    def raw_interpolation_order(self) -> int:
        return self._raw_interpolation_order

    @property
    def label_interpolation_order(self) -> int:
        return self._label_interpolation_order

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def matrix(self) -> np.ndarray:
        """
        Transformation matrix for 'pull-resampling' in affine transformation.
        Given an output voxel index vector `o`, the voxel value is
        determined from the input volume at position `np.dot(matrix, o)`
        """
        return self._matrix
    
    @property
    def inverse_matrix(self) -> np.ndarray:
        """Transformation matrix for 'push-resampling' in affine transformation."""
        return self._inverse_matrix
    

    def configuration_dict(self) -> dict:
        """Return the rescaler configuration as a Python dictionary""" 
        config = {
            'original_voxel_sizes' : self.original_voxel_sizes,
            'rescaled_voxel_sizes' : self.rescaled_voxel_sizes,
            'interpolation_order' : self.interpolation_order,
            'mode' : self.mode
        }
        return config



class DummyRescaler(Rescaler):
    """
    Dummy rescaler does not rescale anything :-)
    """
    def __init__(self) -> None:
        super().__init__(original_voxel_size=1, rescaled_voxel_size=1,
                         interpolation_order=0, mode=None, dtype=None, device='cpu')
    
    def rescale_volume(self, array: np.ndarray) -> np.ndarray:
        return array

    def rescale_label_volume(self, array: np.ndarray) -> np.ndarray:
        return array

    def rescale_general_metadata(self, metadata: dict) -> dict:
        return metadata
    
    def rescale_ijk_position(self, position: np.ndarray) -> np.ndarray:
        return position

    def rescale_landmarks(self, landmark_dicts: Sequence[Dict]) -> Sequence[Dict]:
        return landmark_dicts

    def rescale_segment_infos(self, infos: dict) -> dict:
        return infos

    def rescale_segment_metadata(self, metadata: dict) -> dict:
        return metadata