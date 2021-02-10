import io
import pathlib
import collections
import zipfile as zpf
import numpy as np

import nrrd

from typing import Union, Dict, List, Any, Tuple, Callable, Iterable
from PIL import Image

from reader.tagged_data import RawData, SegmentationData

PathLike = Union[str, pathlib.Path]
ZipMember = Union[str, zpf.ZipInfo]



class MRBFile(zpf.ZipFile):
    """
    Encapsulates/exposes access to the MRB file via default
    CPython standard library ZipFile package.

    Parameters
    ----------

    filepath : PathLike
        The path pointing to the MRB file.
    
    mode : str, optional
        Access mode parameter.
        'r' for reading of an existing file,
        'a' for appending to an existing file,
        'w' to truncate and write a new file,
        'x' for exclusive write to a new file
            (raising FileExistsError if `filepath` preexists)
        Defaults to 'r'.

    compression : int, optional
        Numerical compression constants defined by the zipfile package.
        ZIP compression method used when writing to the archive.
        Should be ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2 or ZIP_LZMA
        Defaults to ZIP_STORED
    
    allowZip64 : bool, optional
        Switches usage of the ZIP64 extension that enables archives
        with sizes > 4 GB. Defaults to True.
    """

    def __init__(self, filepath: PathLike,
                 mode: str = 'r', compression: str = zpf.ZIP_STORED,
                 allowZip64: bool = True) -> None:
        
        super(MRBFile, self).__init__(file=filepath, mode=mode,
                                      compression=compression,
                                      allowZip64=allowZip64)

        if not isinstance(filepath, pathlib.Path):
            filepath = pathlib.Path(filepath)

        assert filepath.is_file(), f'MRB file at {filepath.resolve()} not found!'
        self.filepath = filepath
        
        data_members = self.get_member_info()
        self.raw_members = data_members['raw']
        self.segmentation_members = data_members['seg']


    def get_member_info(self):
        """
        Get the dict of ZipInfo instances that describe data members of the ZipFile.
        The members are subdivided between raw data members and segmentation members
        based on file suffix.

        Example:
        'raw' : [<ZipInfo_raw_1>, <ZipInfo_raw_2>, ...]
        'seg' : [<ZipInfo_seg_1>, <ZipInfo_seg_2>, ...]


        Returns
        -------

        data_members : Dict[str, List]
            Dict with keys 'raw' and 'seg' that hold
            the member lists as values.
            The member lists consist of zipfile.ZipInfo objects.
        """
        segmentation_members = []
        raw_members = []

        for zinfo in self.infolist():
            if zinfo.filename.endswith('.seg.nrrd'):
                segmentation_members.append(zinfo)
            elif zinfo.filename.endswith('.nrrd'):
                raw_members.append(zinfo)

        return {'raw' : raw_members, 'seg' : segmentation_members}
    

    def read_raws(self) -> List[RawData]:
        """
        Return list of parsed raw data members of the MRB file.
        From the zip-internal file, the raw data and corresponding
        raw metadata is processed into RawData tagged array objects.

        Returns
        -------

        raws : List[RawData]
            The list of RawData tagged array objects of parsed raw data members.
            RawData attributes: data, metadata
        """
        local_read_fn = self.read_nrrd
        # below raw_datas is list of tuples 
        # (data, metadata) - (np.ndarray, collections.OrderedDict)
        raw_datas = self._read_members(self.raw_members, local_read_fn)
        return [RawData(*elem) for elem in raw_datas]


    
    def read_segmentations(self) -> List[SegmentationData]:
        """
        Return list of segmentation data members of the MRB file.
        From the zip-internal file, the segmentation data and corresponding
        segmentation metadata is processed into Segmentations data tagged
        array instances.

        Returns
        -------

        raws : List[SegmentationData]
            The list of tuples of parsed segmentation data members.
            SegmentationData attributes: data, metadata, infos

        """
        local_read_fn = self.read_nrrd
        segmentation_datas = self._read_members(self.segmentation_members, local_read_fn)
        return [SegmentationData(*elem) for elem in segmentation_datas]
    

    def _read_members(self,
                      members: Iterable[ZipMember],
                      read_fn: Callable[[ZipMember], Tuple]) -> List[Tuple]:
        """
        Read elements from the members iterable with the read_fn callable and
        return results as a list.
        """
        member_data_list =  []
        for member in members:
            member_data = read_fn(member)
            member_data_list.append(member_data)
        return member_data_list

    
    def read_nrrd(self,
                  member: ZipMember) -> Tuple[np.ndarray, collections.OrderedDict]:
        """
        Access, load and transform NRRD file member of the MRB file into
        a numpy ndarray and corresponding header metadata.
        Accessed member is specified via string or ZipInfo instance.

        Parameters
        ----------

        member : str or zipfile.ZipInfo
            Internal path or ZipInfo instance pointing to the member.
        
        Returns
        -------

        (data, header) : tuple
            2-Tuple of the raw data and the NRRD file header.
            The raw data is a numpy.ndarray.
        """
        member_fobj = io.BytesIO(self.read(member))
        header = nrrd.read_header(member_fobj)
        data = nrrd.read_data(header, fh=member_fobj)
        return (data, header)

    
    def read_nii(self,
                 member: ZipMember) -> Tuple[np.ndarray, collections.OrderedDict]:
        """
        NII file read method.
        """ 
        raise NotImplementedError
    
