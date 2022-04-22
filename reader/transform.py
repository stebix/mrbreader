import numpy as np

from typing import List, Dict, Optional, Tuple, Sequence
from collections import namedtuple

import reader.coordinates as crd
from reader.tagged_data import RawData, LabelData

DataTuple = namedtuple(typename='DataTuple', field_names=('raw', 'label', 'landmarks'))

def ijk_align(raw: RawData, label: LabelData,
              landmarks: Optional[List[Dict]] = None) -> DataTuple:
    """
    Align incongruent raw and label volumes in IJK coordinate space.
    """
    # use LPS everywhere
    target_cosys = 'lps'
    # compute transformation matrices
    # raw data
    T_raw_ijk_to_xyz = crd.ijk_to_xyz_matrix_from_metadata(
        metadata=raw.metadata, target_coordsystem=target_cosys
    )
    T_raw_xyz_to_ijk = np.linalg.inv(T_raw_ijk_to_xyz)
    # label data
    T_label_ijk_to_xyz = crd.ijk_to_xyz_matrix_from_metadata(
        metadata=label.base_metadata, target_coordsystem=target_cosys
    )
    T_label_xyz_to_ijk = np.linalg.inv(T_label_ijk_to_xyz)
    # compute edge coordinates in physical XYZ space
    raw_edge_coords_xyz = crd.compute_xyz_edge_coords(
        ijk_edge_coords=crd.compute_ijk_edge_coords(raw.data.shape),
        T_matrix=T_raw_ijk_to_xyz
    )
    label_edge_coords_xyz = crd.compute_xyz_edge_coords(
        ijk_edge_coords=crd.compute_ijk_edge_coords(label.data.shape),
        T_matrix=T_label_ijk_to_xyz
    )
    # transform edge coordinates into voxel space coordinates
    # of the 'antagonist' 
    raw_edge_coords_in_label_ijk = crd.transform_vectors(raw_edge_coords_xyz, T_label_xyz_to_ijk)
    label_edge_coords_in_raw_ijk = crd.transform_vectors(label_edge_coords_xyz, T_raw_xyz_to_ijk)
    # compute the axis slices that crop the volume towards the
    # common, overlapping sub-volume
    raw_crop_slices_float = crd.compute_axis_slices(label_edge_coords_in_raw_ijk)
    label_crop_slices_float = crd.compute_axis_slices(raw_edge_coords_in_label_ijk)
    # recast to integer slices and compute difference error
    raw_crop_slices, rcs_err = crd.as_int_slices(raw_crop_slices_float, True)
    label_crop_slices, lcs_err = crd.as_int_slices(label_crop_slices_float, True)

    # landmark IJK positions are tied to the initial raw data shape
    # and may have to be transformed
    if landmarks is not None:
        landmarks = crd.recompute_ijk_positions(raw_crop_slices, landmarks)

    return DataTuple(raw=raw.data[raw_crop_slices])
















