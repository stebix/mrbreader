"""
Utility functions adn tools for HDF5 file operations
"""
import numpy as np

from pathlib import PurePosixPath
from typing import Optional, Sequence, Union

import h5py



def create_dataset_from_path(path: PurePosixPath, handle: h5py.File,
                             data: Optional[np.ndarray]) -> h5py.Dataset:
    """
    Create a HDF5 dataset directly from a path instance that
    at least has to specify two parts, i.e. a group and a dataset.
    Multiple intermediate groups are automatically created if
    they are not preexisting. 
    
    Parameters
    ----------
    
    path : PurePosixPath
        The internal path pointing to the dataset. The last
        component is interpreted as the dataset name.
        
    handle : h5py.File
        The open and writable `h5py.File` instance.
        
    data : np.ndarray or None
        The data content of the HDF5 dataset. If data is
        None, then an dummy `h5py.Empty(dtype='f')` is utilized.
        
    
    Returns
    -------
    
    dataset : h5py.Dataset
        The created or retrieved dataset instance.
    """
    if data is None:
        data = h5py.Empty(dtype='f')
    groupnames = path.parts[:-1]
    datasetname = path.parts[-1]
    # start with root group
    current_group = handle['/']
    for groupname in groupnames:
        try:
            group = current_group.require_group(groupname)
        except TypeError as e:
            message = (f'WTF, something terrible has happened during the require_group '
                       f'method of the group {current_group} with the argument: {groupname}')
            raise RuntimeError(message) from e
        current_group = group
    # create dataset after final group fetch/create operation
    dataset = group.create_dataset(datasetname, data=data)
    return  dataset



def create_groups_from_path(path: PurePosixPath, handle: h5py.File) -> h5py.Group:
    """
    Create group or series of groups (implicitly) from path instance.
    If all groups or parts of them are preexisting, then missing ones are created
    and the final group instance is returned.
    
    Parameters
    ----------
    
    path : PurePosixPath
        The internal path pointing to the dataset. 
        All components are interpreted as (nested) groups.
        
    handle : h5py.File
        The open and writable `h5py.File` instance.
    
    Returns
    -------
    
    group : h5py.Group
        The final created or retrieved group instance.
    """
    # start with root group
    current_group = handle['/']
    for groupname in path.parts:
        try:
            group = current_group.require_group(groupname)
        except TypeError as e:
            message = (f'WTF, something terrible has happened during the require_group '
                       f'method of the group {current_group} with the argument: {groupname}')
            raise RuntimeError(message) from e
        current_group = group
    return group


def generate_internal_path(stem: str, i: int) -> PurePosixPath:
    """
    Generate generic internal path for HDF5 files that usually conform
    to the format 'raw/raw-0' and 'label/label-0' for raw volume and label
    data respectively.
    """
    return PurePosixPath(stem, f'{stem}-{i}')



def write_to_attrs(file_element: Union[h5py.Group, h5py.Dataset],
                   dictionary: dict) -> None:
    """Write a dictionary to a HDF5 file element (i.e. group or dataset)"""
    for key, value in dictionary.items():
        file_element.attrs[key] = value



def bulk_write_to_attrs(file_element: Union[h5py.Group, h5py.Dataset],
                        dictionaries: Sequence[dict]) -> None:
    """
    Write a sequence of dictionaries to a HDF5 file element
    (i.e. group or dataset). If identical keys exist, overwriting
    may occur.
    """
    for dictionary in dictionaries:
        write_to_attrs(file_element, dictionary)