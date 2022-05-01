import io
import pathlib
import textwrap
import collections
import json
import warnings
import zipfile as zpf
import numpy as np

import nrrd

from typing import Union, Dict, List, Any, Tuple, Callable, Iterable
from PIL import Image

from reader.tagged_data import RawData, LabelData, WeightData
from reader.utils import (expand_to_4D, reduce_from_4D,
                          ras_to_ijk_matrix, lps_to_ijk_matrix)
from reader.mkparser import extract_fiducial_markups, fit_to_template

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
        self.landmark_members = data_members['lmrk']

    
    def __str__(self) -> str:
        pretty_str = (f'MRBFile(filepath={self.filepath.resolve()}, '
                      f'{len(self.raw_members)} raw members, '
                      f'{len(self.segmentation_members)} segment members, '
                      f'{len(self.landmark_members)} landmark members)')
        return pretty_str
        
    
    def __repr__(self) -> str:
        base_repr = f"MRBFile(filepath='{self.filepath.resolve()}', mode='{self.mode}')"
        raws_repr = '\n'.join(
            (f'Raw members ({len(self.raw_members)} total)', self.pretty_string(self.raw_members))
        )
        seg_repr = '\n'.join(
            (f'Segment members ({len(self.segmentation_members)} total)', self.pretty_string(self.segmentation_members))
        )
        lmrk_repr = '\n'.join(
            (f'Landmark members ({len(self.landmark_members)} total)', self.pretty_string(self.landmark_members))
        )
        member_repr = '\n'.join((raws_repr, seg_repr, lmrk_repr))
        return '\n'.join((base_repr, member_repr))




    def get_member_info(self):
        """
        Get the dict of ZipInfo instances that describe data members of the ZipFile.
        The members are subdivided between raw data members, segmentation members
        and landmark members based on file suffix.

        Example:
        'raw' : [<ZipInfo_raw_1>, <ZipInfo_raw_2>, ...]
        'seg' : [<ZipInfo_seg_1>, <ZipInfo_seg_2>, ...]
        'lmrk' : [<ZipInfo_seg_1>, <ZipInfo_seg_2>, ...]



        Returns
        -------

        data_members : Dict[str, List]
            Dict with keys 'raw', 'seg' and 'lmrk' that hold
            the member lists as values.
            The member lists consist of zipfile.ZipInfo objects.
        """
        segmentation_members = []
        raw_members = []
        landmark_members = []

        for zinfo in self.infolist():
            if zinfo.filename.endswith('.seg.nrrd'):
                segmentation_members.append(zinfo)
            elif zinfo.filename.endswith('.nrrd'):
                raw_members.append(zinfo)
            elif zinfo.filename.endswith('.mrk.json'):
                landmark_members.append(zinfo)

        return {'raw' : raw_members,
                'seg' : segmentation_members,
                'lmrk' : landmark_members}
    

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
    

    def read_stringmatched_raws(self, matchstring: str) -> List[RawData]:
        """
        Read and return raw data members of the MRB file whose (internal)
        filename contains the indicated matchstring.
        """
        local_read_fn = self.read_nrrd
        matching_members = []
        for raw_zipinfo in self.raw_members:
            if matchstring in raw_zipinfo.filename:
                matching_members.append(raw_zipinfo)

        matching_raw_data = self._read_members(matching_members, local_read_fn)
        return [RawData(*elem) for elem in matching_raw_data]
    

    def _read_segmentations(self) -> List:
        """
        Return list of segmentation data members read via NRRD without casting
        them as `LabelData` instances. 
        """
        return self._read_members(self.segmentation_members, self.read_nrrd)

    
    def read_segmentations(self) -> List[LabelData]:
        """
        Return list of segmentation data members of the MRB file.
        From the zip-internal file, the segmentation data and corresponding
        segmentation metadata is processed into Segmentations data tagged
        array instances.

        Returns
        -------

        raws : List[LabelData]
            The list of tuples of parsed segmentation data members.
            LabelData attributes: data, metadata, infos

        """
        return [LabelData(*elem) for elem in self._read_segmentations()]

    
    def read_weights(self) -> List[WeightData]:
        """Read weight data. Currently placeholder."""
        return []
    

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

    
    def read_landmarks(self) -> List[Dict]:
        """
        Read and parse markup data from a JSON-typed MRBFile member.
        """
        landmarks = []
        for lmrk_member in self.landmark_members:
            landmark_dicts = self._read_landmark_member(lmrk_member)
            landmarks.extend(landmark_dicts)
        return landmarks

    
    def _read_landmark_member(self, member: ZipMember) -> List[Dict]:
        """
        Read and process a landmark member.
        Loads the raw data JSON from the MRB/ZIP file and
        parses its content into a list of landmark information dicts.
        IJK position information is added via the transformation deduced from the
        `RawData` object metadata.
        
        Parameters
        ----------

        member: ZipMember
            The ZIP-internal path or info object that is to be loaded.
        
        Returns
        -------

        landmark_dicts: List[Dict]
            The parsed landmark information
        """
        if len(self.raw_members) > 1:
            warnings.warn(
                ('MRBFile has more than one raw member! '
                 'Transformation RAS/LPS to IJK is deduced from first member. '
                 'If metadata is incongruent, faulty results may occur.')
            )
        # load first raw member to access spatial coordinate system information
        raw_0 = RawData(*self._read_members(self.raw_members, read_fn=self.read_nrrd)[0])
        space_kwargs = {
            'space_directions' : raw_0.metadata['space directions'],
            'space_origin' : raw_0.metadata['space origin']
        }
        # load the landmark data from MRB Zipfile
        landmark_json = io.BytesIO(self.read(member))
        landmark_dicts = extract_fiducial_markups(json.load(landmark_json))
        # include IJK coordinate positional information
        for element in landmark_dicts:
            # transformation matrix depends on 3D basis definition
            if element['coordsys'] == 'LPS':
                transform_mat = lps_to_ijk_matrix(**space_kwargs)
            elif element['coordsys'] == 'RAS':
                transform_mat = ras_to_ijk_matrix(**space_kwargs)
            else:
                raise RuntimeError(
                    f'Invalid coordinate system specification: < {element["coordsys"]} >'
                )
            # XYZ position in homogenous coordinates. Hint: JSON package does not cast to np.ndarray
            xyz_pos_hom_cord = expand_to_4D(np.array(element['xyz_position']))
            ijk_position = reduce_from_4D(transform_mat @ xyz_pos_hom_cord)
            element['ijk_position'] = np.rint(ijk_position).astype(np.int32)
            # transform dictionary to match default template for `id` and `label`
            fit_to_template(element)

        return landmark_dicts

    

    @staticmethod
    def pretty_string(obj_coll: List[Any]) -> str:
        """
        Build an indented string representation of the objects in the
        given list.
        """
        pretty_str = '\n'.join(
            (str(obj) for obj in obj_coll)
        )
        return textwrap.indent(pretty_str, prefix='  - ')
