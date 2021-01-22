# from __future__ import annotations  ->  no son, it's Python 3.6 :(
import re
import collections
import numpy as np

from typing import Dict, List, Tuple, Iterable, Any, Union
from utils import relabel

"""
The parsing of files originating from the manual segmentation process
comes with a lot of metadata processing.
"""


class TaggedData:
    """
    Minimal wrapper class for numpy.ndarrays living in the data attribute
    and heterogenous metadata (mostly dictionaries) living in the
    metadata attribute.
    """
    long_repr = False

    def __init__(self, data: np.ndarray, metadata: Any) -> None:
        self.data = data
        self.metadata = metadata
    

    def __repr__(self) -> str:
        """
        Provide a nice instance representation string.
        The length/degree of detail can be switched via the class variable
        `long_repr`.
        """
        if self.long_repr:
            attr_repr = 'arr_shape={}, arr_dtype={}, metadata={}'.format(self.data.shape,
                                                                         self.data.dtype,
                                                                         self.metadata)
        else:
            attr_repr = 'arr_shape={}, arr_dtype={}, metadata_type={}'.format(self.data.shape,
                                                                              self.data.dtype,
                                                                              type(self.metadata))
        repr_str = '{}({})'.format(
            self.TaggedData.__name__,
            attr_repr
        )
        return repr_str


class RawData(TaggedData):
    """
    Tagged CT raw data.
    Usually just a 3D scalar NumPy array (data) and an OrderedDict (metadata).
    """
    pass


class SegmentationData(TaggedData):
    """
    Tagged segmentation data.

    The incoming metadata is expected to be an collections.OrderedDict that holds fields
    that adhere to the 3DSlicer segmentation node specification:
    -> https://apidocs.slicer.org/master/classvtkMRMLSegmentationStorageNode.html#details

    """

    def __init__(self, data: np.ndarray, metadata: Any) -> None:
        super(SegmentationData, self).__init__(data, metadata)

        # creates the per-segment metadata information as `SegmentInfo` instances
        # the attribute is layed out as a dict with the label_value of the
        # segment as the key and the `SegmentInfo` as the value
        self.infos = {
            si.label_value : si for si in SegmentInfo.from_header(metadata)
        }
        # TODO: this is sloooow due to numpy.unique!
        # self._check_consistency()
    

    def _check_consistency(self) -> None:
        """Run some sanity checks that metadata and numerical data are consistent"""
        lbl_vals_from_metadata = set(self.infos.keys())
        # background with label_value 0 is generally not included in metadata dict
        lbl_vals_from_metadata.add(0)
        lbl_vals_from_data = set(np.unique(self.data))
        # TODO: check if numerical datatype shenanigans ruin the day
        # i.e. something along the lines of 1.0 != 1
        symm_diff = lbl_vals_from_data ^ lbl_vals_from_metadata

        if len(symm_diff) != 0:
            msg = (f'Label mismatch between data and metadata! Expected vanishing '
                   f'symmetric difference but got: {symm_diff}')
            raise ValueError(msg)
    

    def __getitem__(self, label_value: int) -> 'SegmentInfo':
        """
        Directly retrieve the segment label information (as a SegmentInfo object)
        via its integer label value that is used as a index here.
        """
        return self.infos[label_value]

    
    def __repr__(self) -> str:
        arr_info_str = 'arr_shape={}, arr_dtype={}'.format(self.data.shape,
                                                         self.data.dtype)
        segment_info = []
        for seginfo_elem in self.infos.values():
            segment_info.append(
                '({}, {})'.format(seginfo_elem.name, seginfo_elem.label_value)
            )        
        seg_info_str = 'segment_count={}, segments=[{}]'.format(
            len(segment_info), ', '.join(segment_info)
        )
        repr_str = '{}({})'.format(self.__class__.__name__,
                                   ', '.join((arr_info_str, seg_info_str)))
        return repr_str

    
    def relabel(self, old: int, new: int) -> None:
        """
        Change the integer label value of a segment.
        The new label must not be an existing label.
        Via this method, the change is reflected in the data array and the
        metadata attribute dict.

        Parameters
        ----------

        old : int
            The old label value.

        new : int
            The corresponding new label value.
        """
        if not (isinstance(old, int) and isinstance(new, int)):
            msg = f'Expecting integer arguments, got {type(old)} and {type(new)}!'
            raise ValueError(msg)
        if new in set(self.infos.keys()):
            msg = f'New label  < {new} > is in existing labels {set(self.infos.keys())}!'
            raise ValueError(msg)
        # modify corresponding SegmentInfo object
        seginfo = self.infos[old]
        seginfo.label_value = new
        # modify array data
        self.data = relabel(self.data, old, new)
        # propagate state changes
        self._update_state_from_infos()


    def swaplabel(self, label_a: int, label_b: int) -> None:
        """
        Swap the integer labels of two segments.

        Parameters
        ----------

        label_a : int
            The first label: is swapped to label_b
        
        label_b : int
            The second label: is swapped to label_a
        """
        if not (isinstance(label_a, int) and isinstance(label_b, int)):
            msg = f'Expecting integer arguments, got {type(label_a)} and {type(label_b)}!'
            raise ValueError(msg)

        self.infos[label_a].label_value = label_b
        self.infos[label_b].label_value = label_a
        labels = [label_a, label_b]
        # modify array data
        self.data = relabel(self.data, labels, reversed(labels))
        # propagate state changes
        self._update_state_from_infos()
        

    def rename(self, label_value: int, new_name: str) -> None:
        """
        Change the name attribute of a segment. The segment is accessed via its
        `label_value`. 

        Parameters
        ----------

        label_value : int
            The integer label value of the segment we want to rename.

        new_name : str
            The new name of the segment. 
        """
        seginfo = self.infos[label_value]
        seginfo.name = new_name
        # propagate state changes
        self._update_state_from_infos()

    
    def recolor(self, label_value: int, color: Tuple[float, float, float]) -> None:
        """
        Change the color attribute of a segment.

        Parameters
        ----------

        label_value : int
            The integer label value of the segment we want to give
            a new color attribute.
        
        color : 3-tuple of float
            The RGB color tuple. Elements expected to be in [0, 1].
        """
        seginfo = self.infos[label_value]
        seginfo.color = color
        # propagate state changes
        self._update_state_from_infos()


    def _update_state_from_infos(self) -> None:
        """
        Update the `self` instance and the `self.metadata` collections.OrderedDict
        (that is inferred from the raw data header) from the `self.infos` attribute.
        There the `SegmentInfo` instances the nicely describe the individual segments
        are stored.

        We use this to method to ensure internal consitency after an interaction
        with the `SegmentInfo` instances for modifications like renaming,
        relabeling, etc. 
        """
        # update the keys that is the integer label_value of the SegmentInfo
        self.infos = {
            si.label_value : si for si in self.infos.values()
        }
        for idx, seginfo in enumerate(self.infos.values()):
            prefix = f'Segment{idx}_'
            self.metadata.update(
                seginfo.to_dict(keystyle='slicer', prefix=prefix)
            )



class SegmentInfo:
    """
    A dataclass-like object to store the metadata information belonging to
    a segment in a 3D-Slicer generated segmentation.
    """
    slicer_to_internal_alias = {
        'Color' : 'color',
        'ColorAutoGenerated' : 'color_autogenerated',
        'Extent' : 'extent',
        'ID' : 'ID',
        'LabelValue' : 'label_value',
        'Layer' : 'layer',
        'Name' : 'name',
        'NameAutoGenerated' : 'name_autogenerated',
        'Tags' : 'tags'
    }
    # reversed mapping
    internal_to_slicer_alias =  {
        value : key for key, value in slicer_to_internal_alias.items()
    }
    # fields for the __repr__ dunder method
    repr_field_names = ['name', 'label_value', 'color', 'ID']

    def __init__(self, name, color, ID, label_value,
                 extent, color_autogenerated, layer,
                 name_autogenerated, tags) -> None:
        
        self.name = name
        self.color = color
        self.ID = ID
        self.label_value = label_value
        self.extent = extent
        self.color_autogenerated = color_autogenerated
        self.layer = layer
        self.name_autogenerated = name_autogenerated
        self.tags = tags

    
    def __repr__(self) -> str:
        # string repr sequence of selected attributes
        attr_kv_seq = ', '.join(
            ('{}={}'.format(name, getattr(self, name))
            for name in self.repr_field_names)
        )
        repr_str = '{}({})'.format(
            self.__class__.__name__,
            attr_kv_seq
        )
        return repr_str

    
    def __eq__(self, other) -> bool:
        """Dunder comparison method."""
        # TODO: We got to think about the case: When do SegmentInfo
        # objects represent the same thing? When label values are identical?
        # Or when additional fields are equal?
        if not isinstance(other, SegmentInfo):
            return NotImplemented
        else:
            return self.ID == other.ID


    def to_dict(self, keystyle='internal', prefix='') -> Dict:
        """
        Turn/devolve the object into a dictionary (Python primitive).

        Parameters
        ----------

        keystyle : str, optional
            Determines the styling of the keys in the returned dict.
            May be 'internal' (yielding pretty snake_case keys)
            or 'slicer' (yielding ugly CamelCase keys)
            Defaults to 'internal'
        
        prefix : str, optional
            Optional prefix prepended to the keys. Only active for
            'slicer' keystyle. May be used to prepend the Slicer-
            internal 'SegmentN_' to the keys.
            Defaults to '' (empty string)
        
        Returns
        -------

        obj_dict : dict
            The SegmentInfo instance as a Python-primitive
            dictionary
        """
        if keystyle not in ('internal', 'slicer'):
            msg = f'Unrecognized keystyle: {keystyle}'
            raise ValueError(msg)
        
        if keystyle == 'internal':
            obj_dict = {
                key : getattr(self, key)
                for key in self.internal_to_slicer_alias.keys()
            }
        else:
            obj_dict = {
                ''.join((prefix, key)) : getattr(self, value)
                for key, value in self.slicer_to_internal_alias.items()
            }
            # for full symmetry we have to back-convert the color float tuple
            # into a space-separated string
            obj_dict[''.join((prefix, 'Color'))] = ' '.join(str(val) for val in self.color)
        return obj_dict


    @property
    def color(self) -> Tuple[float]:
        return self._color
    

    @color.setter
    def color(self, c: Iterable) -> None:
        """
        Set the segment color. Must be a RGB tuple (r, g, b)
        The new color specification may be a space-separated string
        whose parts are required to be float-castable.
        Otherwise a tree tuple is required.
        """
        if isinstance(c, str):
            # try RGB interpretation from string
            c = c.split(' ')
        
        c_checked = []
        for val in c:
            val = float(val)
            assert val >= 0 and val <= 1, 'RGB value must be in closed interval [0, 1]'
            c_checked.append(val)
        
        self._color = tuple(c_checked)


    @property
    def label_value(self) -> int:
        return self._label_value
    

    @label_value.setter
    def label_value(self, new_label_value: int) -> None:
        """
        Managed attribute setter for label_value, which
        is expected to be an integer.
        """
        if not isinstance(new_label_value, int):
            new_label_value = int(new_label_value)
        
        self._label_value = new_label_value



    @classmethod
    def from_header(cls,
                    header_data: collections.OrderedDict) -> List['SegmentInfo']:
        """
        Directly create the exhaustive list of SegmentInfo instances from
        the header data
        """
        segment_prefix = 'Segment'
        pattern = re.compile(pattern=segment_prefix + '\d')
        segment_attrs = {}
        # exhaustive search over the full header dictionary
        for key, value in header_data.items():
            # only concern ourselves segment key - value pairs
            if pattern.match(key):
                # 3D Slicer key naming scheme: (SegmentN_attrname, attrval)
                segment_id, attr_alias = key.split('_')
                # translate the 3D Slicer attr alias to our preferred internal name
                attr_name = cls.slicer_to_internal_alias[attr_alias]
                # insert into the dictionary
                try:
                    segment_attrs[segment_id]
                except KeyError:
                    # not encountered this segment up until now
                    # make a new subdictionary
                    segment_attrs[segment_id] = {}
                
                segment_attrs[segment_id][attr_name] = value
        
        return [cls(**kwargs) for kwargs in segment_attrs.values()]

    