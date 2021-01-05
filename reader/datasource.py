import pathlib
import logging
import numpy as np
import pydicom

from collections.abc import Iterable
from typing import Union, Dict, List, Tuple


PathLike = Union[str, pathlib.Path]


logger = logging.getLogger(__name__)


class AbstractDataSource:

    def __init__(self):
        pass
    

    def get_raw_volume(self):
        raise NotImplementedError
    

    def get_label_volume(self):
        raise NotImplementedError

    
    def get_weight_volume(self):
        raise NotImplementedError

    
    def get_data_volume(self):
        raise NotImplementedError



class DicomStack(AbstractDataSource):
    """
    Use a stack of DICOM files organized in a nested or flat directory structure
    as the data source.
    """

    def __init__(self, directorypath: PathLike):
        if not isinstance(directorypath, pathlib.Path):
            directorypath = pathlib.Path(directorypath)

        self.directorypath = directorypath


    def is_raw(self, filepath: PathLike) -> bool:
        """
        Check if filename indicates a raw or volume data
        file.
        """
        if not isinstance(filepath, pathlib.Path):
            filepath = pathlib.Path(filepath)
        return self.check_affinity(string=filepath.name, candidate='raw')

    
    def is_label(self, filepath: PathLike) -> bool:
        """
        Check if filename indicates a label or ground-truth
        file.
        """
        if not isinstance(filepath, pathlib.Path):
            filepath = pathlib.Path(filepath)
        return self.check_affinity(string=filepath.name, candidate='label')


    def is_weight(self, filepath: PathLike) -> bool:
        """
        Check if filename indicates a weight data file.
        """
        if not isinstance(filepath, pathlib.Path):
            filepath = pathlib.Path(filepath)
        return self.check_affinity(string=filepath.name, candidate='weight')


    @staticmethod
    def check_affinity(string: str, candidate: str) -> bool:
        """
        Check if the given string (probably a filename) indicates affinity to
        a certain category that is encoded by the candidate string.

        Currently simplistic start or end checking is employed.
        Maybe regex stuff later ...?
        """
        if string.endswith(candidate) or string.endswith(candidate):
            return True
        else:
            return False


    @staticmethod
    def load_stack(filepaths: List[PathLike]) -> np.ndarray:
        """
        Select and load all given DICOM files and stack
        them into a numpy array.
        The DICOM files are stacked along the first axis.

        Parameters
        ----------

        filepaths : Iterable of PathLike
            The filepaths from which the DICOM files
            are pulled.

        Returns
        -------

        image_volume : numpy.ndarray
            The loaded DICOM stack as a numpy array.
        """
        # wrap list to enable single element loading
        if not isinstance(filepaths, Iterable):
            filepaths = [filepaths]

        dicom_suffices = ['dcm', 'DCM', 'DICOM', 'dicom', 'dic']
        dcm_file_dsets = []
        for entry in filepaths:
            if entry.suffix in dicom_suffices:
                try:
                    file_dset = pydicom.filereader.dcmread(entry)
                except (pydicom.InvalidDicomError, TypeError) as e:
                    logger.error((f'Skipping file element at {entry.resolve()}  '
                                  f'Reason: {e}'))
                    continue

                dcm_file_dsets.append(file_dset)

        if dcm_file_dsets is None:
            # somehow none of the given files could be identified/parsed
            # as a valid DICOM file, we communicate this explicitly
            raise RuntimeError('Unable to parse any of the files as DICOM')

        # stack raw voxel data to numpy array 
        image_volume = np.stack(
            [file_dset.convert_pixel_data() for file_dset in dcm_file_dsets],
            axis=0
        )
        return image_volume