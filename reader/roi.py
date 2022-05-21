"""
Excise region/volume of interest from a larger array via a
specification of an extent. 
"""
import numpy as np
import dataclasses

from itertools import chain
from copy import deepcopy
from typing import Sequence, Tuple, Union, List

from reader.utils import extent_string_as_array, extent_string_as_points


@dataclasses.dataclass
class ROISpec:
    """
    Compound information about ROI in a 3D volume arrray.
    """
    axis0: slice
    axis1: slice
    axis2: slice
    original_shape: Tuple[int]
    shape: Tuple[int] = dataclasses.field(init=False)

    def __post_init__(self):
        """Compute shape after initializer method."""
        self.shape = tuple((slc.stop - slc.start for slc in self))


    def __iter__(self):
        return iter((self.axis0, self.axis1, self.axis2))

    
    @property
    def slices(self) -> Tuple[slice]:
        return tuple(self)


    def transform_ROI_to_original(self, ROI_coordinates: Sequence[int]) -> np.ndarray:
        """
        Transform from ROI IJK coordinates to original array IJK coordinates.
        """
        return transform_point_from_ROI(ROI_coordinates, self.slices)

    
    @classmethod
    def from_extent(cls,
                    extent: Union[str, Sequence[int]],
                    pad_width: int,
                    original_shape: Sequence[int]) -> 'ROISpec':
        if isinstance(extent, str):
            extent = extent_string_as_array(extent)
        else:
            extent = np.array(extent)
        if extent.size != 6:
            raise ValueError(f'extent array has {extent.size} elements, expected 6')
        slices = pad_slices(compute_extent_slices(extent), pad_width)
        return cls(*slices, original_shape)

    

class Roifier:
    """Roi-ifies a whole lot of stuff."""

    def __init__(self, roispec: ROISpec) -> None:
        self.roispec = roispec
    

    def transform_original_to_ROI(self, original_coordinates: Sequence[int]) -> np.ndarray:
        """
        Transform from IJK coordinates in the original array to
        in-ROI IJK coordinates
        """
        return transform_point_to_ROI(original_coordinates, self.roispec.slices)


    def roify_volume(self, volume: np.ndarray) -> np.ndarray:
        if volume.ndim != 3:
            raise ValueError(f'expected volumetric array with ndim 3, got ndim {volume.ndim}')
        # Index into array with ROI-selecting slices.
        return volume[self.roispec.axis0, self.roispec.axis1, self.roispec.axis2]


    def roify_raw_metadata(self, metadata: dict) -> dict:
        metadata = deepcopy(metadata)
        original_shape = metadata['sizes']
        if tuple(original_shape) != tuple(self.roispec.original_shape):
            message = (f'Roification data mismatch: attempting to roify metadata with '
                       f'shape {original_shape} and ROISpec shape {self.roispec.original_shape}')
            raise RuntimeError(message)
        metadata['sizes'] = np.array(self.roispec.shape)
        return metadata

    
    def roify_label_general_metadata(self, metadata: dict) -> dict:
        """Label general metadata"""
        return self.roify_raw_metadata(metadata)
    

    def roify_label_segment_metadata(self, segments: dict) -> dict:
        roified_segments = deepcopy(segments)
        for ID, attrdict in segments.items():
            extent = self.roify_extent(extent_string_as_array(attrdict['Extent']), format_='array')
            roified_segments[ID]['Extent'] = extent
        return roified_segments


    def roify_landmark(self, landmark: dict) -> dict:
        landmark = deepcopy(landmark)
        landmark['ijk_position'] = self.transform_original_to_ROI(landmark['ijk_position'])
        return landmark


    def roify_landmarks(self, landmarks: Sequence[dict]) -> List[dict]:
        return [self.roify_landmark(landmark) for landmark in landmarks]

    
    def roify_extent(self, extent: np.ndarray, format_: str) -> Union[str, np.ndarray]:
        if not isinstance(extent, np.ndarray):
            extent = np.array(extent)
        start, stop = extent[0::2], extent[1::2]
        roified_start = self.transform_original_to_ROI(start)
        roified_stop = self.transform_original_to_ROI(stop)
        if format_ == 'string':
            return ' '.join((str(c) for c in chain(*zip(roified_start, roified_stop))))
        elif format_ == 'array':
            return np.array(list(chain(*zip(roified_start, roified_stop))))
        else:
            raise RuntimeError(f'invalid format specification: {format_}')




        


def transform_coordinate_to_ROI(coordinate: int, slc: slice) -> int:
    if coordinate < slc.start or coordinate > slc.stop:
        raise ValueError(f'coordinate {coordinate} out of bounds for {slc}')
    return coordinate - slc.start


def transform_coordinate_from_ROI(coordinate: int, slc: slice) -> int:
    size = slc.stop - slc.start
    if coordinate >= size:
        raise ValueError(f'coordinate {coordinate} out of bounds for {slc}')
    return coordinate + slc.start


def transform_point_to_ROI(point: Sequence[int], slices: Sequence[slice]) -> np.ndarray:
    """
    Transform IJK point into coordinates inside/relative to ROI. 
    """
    point = np.array(point)
    if point.size > 3:
        raise ValueError(f'expected 3 coordinates, got {point.size}')
    slices = list(slices)
    if not isinstance(slices, (tuple, list)):
        slices = list(slices)
    if len(slices) > 3:
        raise  ValueError('only three slices allowed for ROI specification')
    return np.array([transform_coordinate_to_ROI(c, s) for c, s in zip(point, slices)])


def transform_point_from_ROI(point: Sequence[int], slices: Sequence[slice]) -> np.ndarray:
    """
    Transform ROI IJK point into coordinates outside of ROI.
    """
    point = np.array(point)
    if point.size > 3:
        raise ValueError(f'expected 3 coordinates, got {point.size}')
    slices = list(slices)
    if not isinstance(slices, (tuple, list)):
        slices = list(slices)
    if len(slices) > 3:
        raise  ValueError('only three slices allowed for ROI specification')
    return np.array([transform_coordinate_from_ROI(c, s) for c, s in zip(point, slices)])

    

def compute_extent_slices(extent: Sequence[int]) -> Tuple[slice]:
    """Compute the per-axis slices from an extent specification."""
    if not isinstance(extent, np.ndarray):
        extent = np.array(extent)
    slices = []
    for start, stop in zip(extent[0::2], extent[1::2]):
        slices.append(slice(start, stop))
    return tuple(slices)



def pad_slices(slices: Sequence[slice], pad_width: int) -> List[slice]:
    """Pre- and post pad a sequence of slices with the provided pad_width"""
    padded_slices = []
    for slc in slices:
        padded_slices.append(slice(slc.start-pad_width, slc.stop+pad_width))
    return padded_slices


    