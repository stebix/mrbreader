import pathlib
import itertools
import h5py

from typing import Dict, List, Tuple, Union, NewType, Sequence

from reader.tagged_data import RawData, LabelData, WeightData
from reader.mrbfile import MRBFile


"""
This module enables the programmatic exporting of RawData and LabelData
pairs as HDF5 files.
The exporting structure is geared towards direct use of the HDF5 files
as training data input for the segmentation_net project.
"""

PathLike = NewType('PathLike', Union[str, pathlib.Path, None])

# TODO: Refactor this approach ...

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
                 raw_internal_grpname: str = 'raw',
                 raw_internal_dsetname: str = 'raw-',
                 label_internal_grpname: str = 'label',
                 label_internal_dsetname: str = 'label-',
                 weight_internal_grpname: str = 'weight',
                 weight_internal_dsetname: str = 'weight-',
                 landmark_internal_grpname: str = 'landmark',
                 landmark_internal_dsetname: str = 'landmark-',
                 force_write: bool = False,
                 store_metadata: bool = True) -> None:

        # raw data 
        self.raw_internal_grpname = raw_internal_grpname
        self.raw_internal_dsetname = raw_internal_dsetname
        # label data
        self.label_internal_grpname = label_internal_grpname
        self.label_internal_dsetname = label_internal_dsetname
        # weight data
        self.weight_internal_grpname = weight_internal_grpname
        self.weight_internal_dsetname = weight_internal_dsetname
        # landmark data
        self.landmark_internal_grpname = landmark_internal_grpname
        self.landmark_internal_dsetname = landmark_internal_dsetname

        self.force_write = force_write
        self.store_metadata = store_metadata

    

    def store_mrb(self, save_path: PathLike, mrbfile: MRBFile) -> None:
        """
        Export/store a given `MRBFile` object as a HDF5 file on disk to the
        location given by `save_path`.
        """
        save_path = pathlib.Path(save_path)

        if save_path.is_file() and self.force_write:
            hdf5_write_mode = 'w'
        elif save_path.is_file() and not self.force_write:
            err_msg = f'File already existing at location: < {save_path.absolute()} >' 
            raise FileExistsError(err_msg)
        elif save_path.is_dir():
            raise IsADirectoryError(f'Save path points to a directory: {save_path.resolve()}')
        else:
            hdf5_write_mode = 'x'

        # all tagged data extracted from the MRB file
        total_tagged_datas = itertools.zip_longest(
            mrbfile.read_raws(), mrbfile.read_segmentations(), mrbfile.read_weights(),
            fillvalue=None
        )
        # iterate over raw, segmentation and weight and save appropriately
        with h5py.File(save_path, mode=hdf5_write_mode) as wfile:
            for idx, tagged_data_tuple in enumerate(total_tagged_datas):
                internal_path_tuple = self.construct_internal_paths(idx)
                print(tagged_data_tuple)
                for int_path, tgdat in zip(internal_path_tuple, tagged_data_tuple):
                    if tgdat is None:
                        continue
                    # write numerical array data
                    wfile[int_path] = tgdat.data
                    # write metadata
                    if self.store_metadata:
                        for k, v in tgdat.metadata.items():
                            wfile[int_path].attrs[k] = v
        
            # save landmark data in separate group
            # every landmark gets an empty placeholder dataset
            lmrk_grp = wfile.create_group(self.landmark_internal_grpname)
            for idx, landmark in enumerate(mrbfile.read_landmarks()):
                dset_name = ''.join((self.landmark_internal_dsetname, str(idx)))
                dset = lmrk_grp.create_dataset(dset_name, data=h5py.Empty(dtype='f'))
                for k, v in landmark.items():
                    dset.attrs[k] = v
        
        return None

    
    def store(self,
              save_path: PathLike,
              tagged_raw_data: Sequence[RawData],
              tagged_label_data: Sequence[Union[LabelData, None]] = [],
              tagged_weight_data: Sequence[Union[WeightData, None]] = []) -> None:
        """
        Store the various data instances (raw, label and weight) to a single
        HDF5 file specified by the save path. 

        Parameters
        ----------

        save_path : PathLike
            The save path of the HDF5 file. Depending on the selected
            overwriting behaviour, paths to pre-existing files may fail.
        
        tagged_raw_data : Sequence[RawData]
            The raw image data objects (TaggedData).
        
        tagged_label_data : Sequence[LabelData], optional
            The label data objects giving the raw image data voxels
            a semantic class. Defaults to None.

        tagged_weight_data : Sequence[WeightData], optional
            The weight data objects giving the raw image data voxels
            a specific weighting factor. Defaults to None.
        """
        # enforce various type castings
        if not isinstance(save_path, pathlib.Path):
            save_path = pathlib.Path(save_path)

        if save_path.is_file() and self.force_write:
            hdf5_write_mode = 'w'
        elif save_path.is_file() and not self.force_write:
            err_msg = f'File already existing at location: < {save_path.absolute()} >' 
            raise FileExistsError(err_msg)
        else:
            hdf5_write_mode = 'x'
        
        tagged_datas = itertools.zip_longest(
            tagged_raw_data, tagged_label_data, tagged_weight_data,
            fillvalue=None
        )

        with h5py.File(save_path, mode=hdf5_write_mode) as writefile:
            for idx, tagged_data_tuple in enumerate(tagged_datas):
                internal_paths = self.construct_internal_paths(idx)

                # iterate over (raw, label, weight)
                for int_path, tagged_data in zip(internal_paths, tagged_data_tuple):
                    if tagged_data is None:
                        continue
                    # actual numerical data storage
                    writefile[int_path] = tagged_data.data
                    # metadata storage as h5py.File.Group.dataset.attribute
                    if self.store_metadata:
                        for key, value in tagged_data.metadata.items():
                            writefile[int_path].attrs[key] = value


    def construct_internal_paths(self, idx: int) -> Tuple[str, str, str]:
        """
        Construct the tuple of HDF5-internal paths for the array data tuple
        (raw, label, weight)
        -> (raw_internal_path, label_internal_path, weight_internal_path)

        Parameters
        ----------

        idx : int
            The index differentiating the dataset inside the group
            of the HDF5 file.
        
        Returns
        -------

        internal_paths : 3-tuple of string
            The constructed internal paths:
            f'grp_name / dset_name{idx}'
            (raw_internal_path, label_internal_path, weight_internal_path)
        """
        grp_names = [self.raw_internal_grpname,
                     self.label_internal_grpname,
                     self.weight_internal_grpname]
        dset_names = [self.raw_internal_dsetname,
                      self.label_internal_dsetname,
                      self.weight_internal_dsetname]
        internal_paths = []
        for grp_name, dset_name in zip(grp_names, dset_names):
            internal_paths.append('/'.join((grp_name, dset_name + str(idx))))
        return tuple(internal_paths)

        

