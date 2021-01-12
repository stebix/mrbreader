import io
import pathlib
import collections
import zipfile as zpf
import numpy as np

import nrrd

from typing import Union, Dict, List, Any, Tuple
from PIL import Image

PathLike = Union[str, pathlib.Path]


class TaggedData:

    def __init__(self, data: np.ndarray, metadata: Any) -> None:
        self.data = data
        self.metadata = metadata


class RawData(TaggedData):
    pass


class SegmentationData(TaggedData):
    pass




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

    compression : str, optional
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
        
        data_members = self.get_data_members()

        self.raw_members = data_members['raw']
        self.segmentation_members = data_members['seg']


    def get_data_members(self):
        """
        Get the dict of ZipInfo instances that describe data members of the ZipFile.
        The members are subdivided between raw data members and segmentation members
        based on file suffix.

        Example:
        'raw' : [<ZipInfo_raw_1>, <ZipInfo_raw_2>, ...]
        'seg' : [<ZipInfo_seg_1>, <ZipInfo_seg_2>, ...]


        Returns
        -------

        data_members : Dict
            Dict with keys 'raw' and 'seg' that hold
            the member lists as values.
        """
        segmentation_members = []
        raw_members = []

        for zinfo in self.infolist():
            if zinfo.filename.endswith('.seg.nrrd'):
                segmentation_members.append(zinfo)
            elif zinfo.filename.endswith('.nrrd'):
                raw_members.append(zinfo)

        return {'raw' : raw_members, 'seg' : segmentation_members}
    

    def get_raws(self) -> List[Tuple]:
        """
        Return list of raw
        """
        raws = []
        for raw_member in self.raw_members:
            tagged_raw_data = self.read_nrrd(raw_member)
            raws.append(tagged_raw_data)

        return raws

    
    def get_segmentations(self) -> List[Tuple]:
        """
        Return list of segmentations
        """
        segmentations = []
        for seg_member in self.segmentation_members:
            tagged_seg_data = self.read_nrrd(seg_member)
            segmentations.append(tagged_seg_data)
            
        return segmentations
    

    def read_nrrd(self,
                  member: Union[str, zpf.ZipInfo]) -> Tuple[np.ndarray, collections.OrderedDict]:
        """
        Access, load and transform NRRD file member of the MRB file into
        a numpy ndarray.
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

        

    





class MRBReader:
    
    def __init__(self, filepath: PathLike):
        self.filepath = filepath
        self.file = None

    def print_content(self):
        pass


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    testfile_path = pathlib.Path('C:/Users/Jannik/Desktop/mrbreader/tests/assets/mu_ct_seg.mrb')
    assert testfile_path.is_file()


    testmrb = MRBFile(testfile_path)

    data, metadata = testmrb.get_segmentations()[0]

    print(data.shape)
    print(metadata)

    # with zpf.ZipFile(testfile_path, mode='r') as f:
    #     print(f.namelist())

    #     for item in f.infolist():
    #         print(item.filename)
    #         if item.filename.endswith('.png'):

    #             # img_data = Image.frombytes(f.read(item))

    #             img_data = Image.open(io.BytesIO(f.read(item)))

    #             img_data = np.asarray(img_data)


    #             print(f'IMG DATA SHAPE: {img_data.shape}')

    #             fig, ax = plt.subplots()
    #             ax.imshow(img_data)

    #             plt.show()


