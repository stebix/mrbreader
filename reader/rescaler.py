import numpy as np

from copy import deepcopy
from typing import Sequence, Dict

from reader.rescaletools import (rescale,
                                 create_rescaling_matrix,
                                 transform_ijk_position)

class Rescaler:

    def __init__(self,
                 original_voxel_size: float,
                 rescaled_voxel_size: float,
                 interpolation_order: int) -> None:
        self._original_voxel_size = original_voxel_size
        self._rescaled_voxel_size = rescaled_voxel_size
        self._interpolation_order = interpolation_order
        self._rescaling_factor = rescaled_voxel_size / original_voxel_size
        self._transformation_matrix = create_rescaling_matrix(
            original_voxel_size=original_voxel_size,
            rescaled_voxel_size=rescaled_voxel_size
        )

    
    def rescale_volume(self, array: np.ndarray) -> np.ndarray:
        """
        Rescale a 3D volume using the set original and rescaled voxel sizes.
        """
        assert array.ndim == 3, f'expected 3D array, got ndim = {array.ndim}'
        return rescale(array, self.original_voxel_size, self.rescaled_voxel_size,
                       self.interpolation_order)
    

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
                {key : transform_ijk_position(landmark_dict[key], self.transformation_matrix)}
            )
            rescaled_landmark_dicts.append(modified_landmark_dict)
        return rescaled_landmark_dicts


    @property
    def original_voxel_size(self) -> float:
        return self._original_voxel_size
    
    @property
    def rescaled_voxel_size(self) -> float:
        return self._rescaled_voxel_size
    
    @property
    def rescaling_factor(self) -> float:
        return self._rescaling_factor

    @property
    def interpolation_order(self) -> int:
        return self._interpolation_order

    @property
    def transformation_matrix(self) -> np.ndarray:
        return self._transformation_matrix