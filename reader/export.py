import pathlib
import h5py

from typing import Dict, List, Tuple, Union, NewType

from reader.tagged_data import RawData, SegmentationData, WeightData


"""
This module enables the programmatic exporting of RawData and SegmentationData
pairs as HDF5 files.
The exporting structure is geared towards direct use of the HDF5 files
as training data input for the segmentation_net project.
"""

PathLike = NewType('PathLike', Union[str, pathlib.Path, None])

class HDF5Exporter:
    """
    Bundled functionality to save raw, label and weight data and subsets thereof
    to a single HDF5 file.

    Parameters
    ----------

    raw_internal_path : PathLike
        The HDF5 internal path of the raw data.
    
    label_internal_path : PathLike, optional
        The HDF5 internal path of the label data.
        Must be set if we want to save label data with
        exporter instance. Defaults to None.

    weight_internal_path : PathLike, optional
        The HDF5 internal path of the weight data.
        Must be set if we want to save weight data with
        exporter instance. Defaults to None.

    force_write : bool, optional
        Sets the overwriting behaviour of the exporter instance.
        If True, the output file will be overwritten.
        Default to False.
    
    store_metadata : bool, optional
        Sets the storing of metadata as HDF5 dataset attributes.
        Defaults to True.
    """

    def __init__(self,
                 raw_internal_path: PathLike,
                 label_internal_path: PathLike = None,
                 weight_internal_path: PathLike = None,
                 force_write: bool = False,
                 store_metadata: bool = True) -> None:
    
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path
        self.weight_internal_path = weight_internal_path
        self.force_write = force_write
        self.store_metadata = store_metadata


    
    def store(self,
              save_path: PathLike,
              tagged_raw_data: RawData,
              tagged_label_data: Union[SegmentationData, None] = None,
              tagged_weight_data: Union[WeightData, None] = None) -> None:
        """
        Store the various data instances (raw, label and weight) to a single
        HDF5 file specified by the save path. 

        Parameters
        ----------

        save_path : PathLike
            The save path of the HDF5 file. Depending on the selected
            overwriting behaviour, paths to pre-existing files may fail.
        
        tagged_raw_data : RawData
            The raw image data object (TaggedData).
        
        tagged_label_data : SegmentationData, optional
            The label data giving the raw image data voxels
            a semantic class. Defaults to None.

        tagged_weight_data : WeightData, optional
            The weight data giving the raw image data voxels
            a specific weighting factor. Defaults to None.
        """
        if not isinstance(save_path, pathlib.Path):
            save_path = pathlib.Path(save_path)

        if save_path.is_file() and self.force_write:
            hdf5_write_mode = 'w'
        elif save_path.is_file() and not self.force_write:
            err_msg = f'File already existing at location: < {save_path.absolute()} >' 
            raise FileExistsError(err_msg)
        else:
            hdf5_write_mode = 'x'
        
        tagged_datas = [tagged_raw_data, tagged_label_data, tagged_weight_data]
        internal_paths = [self.raw_internal_path, self.label_internal_path,
                          self.weight_internal_path]

        with h5py.File(save_path, mode=hdf5_write_mode) as writefile:
            for tagged_data, internal_path in zip(tagged_datas, internal_paths):
                if tagged_data is None:
                    continue
                else:
                    assert_msg = (f'Missing the HDF5 internal path for the given data '
                                  f'object {str(tagged_data)}!')
                    assert internal_path is not None, assert_msg
                    
                # actual numerical data storage
                writefile[internal_path] = tagged_data.data
                # metadata storage as h5py.File.Group.dataset.attribute
                if self.store_metadata:
                    for key, value in tagged_data.metadata.items():
                        writefile[internal_path].attrs[key] = value


