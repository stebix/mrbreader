"""
Provide functionality to rearrange Sieber et al. label data from
many-class, multi-layer label data towards clean, single class
{scala tympani, scala vestibuli}
"""
import collections
import numpy as np

from copy import deepcopy
from typing import Tuple, Sequence, Dict



class InvalidSegmentKey(Exception):
    """Header segment-specific dictionary key does not adhere to expected layout standards"""
    pass


SegmentKey = collections.namedtuple('SegmentKey', ['ID', 'attrname'])


def parse_segment_key(key: str) -> SegmentKey:
    """Split header segment key string into semantic parts"""
    try:
        segment, attrname = key.split('_')
    except ValueError as e:
        raise InvalidSegmentKey(f'Failed to split key <{key}> into segment part and attrname') from e
    
    if not segment.startswith('Segment'):
        raise InvalidSegmentKey(f'Key {key} has no <Segment> prefix')
    
    try:
        # From 3DSlicer the key is given by `SegmentN` where N is an int.
        ID = int(segment[len('Segment'):])
    except ValueError as e:
        raise InvalidSegmentKey(f'Integer ID unintelligible from segment key: {segment}') from e
    
    return SegmentKey(ID, attrname)
    
    

def parse_header(header: dict) -> Tuple[dict]:
    """
    Parse a flat header OrderedDict into a tuple of a
        1:  nested dictionary of the layout segment_ID -> {infodict}
        2:  flat dictionary with general information
        
    Parameters
    ----------
    
    header :  dict
        The header as a mapping-type object.
    
    Returns
    -------
    
    (segments, generals) : Tuple of dict
        2-tuple of information dictionaries.    
    """
    non_segment_keys = set()
    segments = collections.defaultdict(dict)
    for key, value in header.items():
        try:
            parsed_key = parse_segment_key(key)
        except InvalidSegmentKey:
            non_segment_keys.add(key)
        else:
            segments[parsed_key.ID][parsed_key.attrname] = value
    # general information not belonging to specific segments
    generals = {key : header[key] for key in non_segment_keys}
    return (segments, generals)
            

def is_valid_segments_dict(segments: dict) -> tuple:
    """Check validity of segments dictionary by searching for non-unique label values or ID strings"""
    segment_count = len(segments)
    label_values = set()
    ID_strings = set()
    names = set()
    invalid_reason = None
    for attrdict in segments.values():
        label_values.add(attrdict['LabelValue'])
        ID_strings.add(attrdict['ID'])
        names.add(attrdict['Name'])
    if len(label_values) != segment_count:
        # raise ValueError(f'Segments dict defines {len(label_values)} for {segment_count} segments')
        invalid_reason = f'Segments dict defines {len(label_values)} label values for {segment_count} segments'
        return (False, invalid_reason)
    if len(ID_strings) != segment_count:
        # raise ValueError(f'Segments dict defines {len(ID_strings)} for {segment_count} segments')
        invalid_reason = f'Segments dict defines {len(ID_strings)} ID strings for {segment_count} segments'
        return (False, invalid_reason)
    if len(names) != segment_count:
        invalid_reason = f'Segments dict has non-unique names for the segments'
        return (False, invalid_reason) 
    return (True, invalid_reason)


def is_multilayer_segments_dict(segments: dict) -> bool:
    """Check if segments dictionary defines multiple layers."""
    unique_layers = set()
    for attrdict in segments.values():
        unique_layers.add(attrdict['Layer'])
    if len(unique_layers) > 1:
        return True
    return False
        

def group_by_layer(segments: dict) -> dict:
    """
    Group the segments in the dict by the layer.
    This forms the nested mapping: 
        layer -> {ID -> attrdict}
    """
    by_layer = collections.defaultdict(dict)
    for ID, attrdict in segments.items():
        layer = int(attrdict['Layer'])
        by_layer[layer][ID] = attrdict
    return by_layer


def select_by_names(segments: dict, desired_segment_names: Sequence[str]) -> tuple:
    """
    Create two dicts by selection from segments dict via names:
    First dict contains the desired segments, the second the
    irrelevant, non-selected segments.
    The `segments` dict is expected to adhere to the {ID -> infodict} mapping
    scheme.
    """
    # We do not want to modify the provided dictionary in-place.
    segments = deepcopy(segments)
    name_to_ID = {attrdict['Name'] : ID for ID, attrdict in segments.items()}
    selected = {}
    for name in desired_segment_names:
        try:
            ID = name_to_ID[name]
        except KeyError:
            message = (f'Segment name "{name}" not found. Offending segments '
                       f'dict: {segments}')
            raise KeyError(message)
        selected[ID] = segments.pop(ID)
    # Recast as standard dict: `segments` is often a defaultdict.
    deselected = dict(segments)
    return (selected, deselected)


def get_old_new_label_pairs(segments: dict, name_to_label_mapping: dict) -> dict:
    """
    Compute the mapping of name to old and new label value pair for
    a given relabeling scheme.
        {name -> (old, new)}
    """
    name_to_oldnew = {}
    name_to_attrdict = {segment['Name'] : segment for segment in segments.values()}
    for name, newlabel in name_to_label_mapping.items():
        oldlabel = name_to_attrdict[name]['LabelValue']
        # If segments dict came directly from NRRD header, `oldlabel` is a string.
        name_to_oldnew[name] = (int(oldlabel), newlabel)
    return name_to_oldnew


def get_unique_label_values(segments: dict) -> set:
    """
    Get the set of unique label values defined by the `segments` mapping of the form
    ID -> attrdict
    """
    label_values = set()
    for attrdict in segments.values():
        label_values.add(int(attrdict['LabelValue']))
    return label_values


def reassign_label_segments_values(segments: dict, name_to_label_mapping: dict) -> dict:
    """
    Reassign label values of segments in the ID -> attrdict mapping following the provided
    name to new label mapping.
    """
    updated_segments = {**segments}
    name_to_ID = {attrdict['Name'] : ID for ID, attrdict in segments.items()}
    for name, new_label_value in name_to_label_mapping.items():
        ID = name_to_ID[name]
        # Utilize deepcopy to avoid in-place modification of `segments` dict argument.
        attrdict = deepcopy(segments[ID])
        attrdict['LabelValue'] = new_label_value
        updated_segments[ID] = attrdict
    return updated_segments


def modify_label_array_values(labelarray: np.ndarray,
                              old_new_pairs: Sequence[tuple]) -> np.ndarray:
    """
    Modify a label array by changing the label values according
    to the sequence of tuples `old_new_pairs`
    """
    modified_labelarray = np.zeros_like(labelarray)
    for (oldval, newval) in old_new_pairs:
        mask = np.where(labelarray == oldval, True, False)
        modified_labelarray[mask] = newval
    return modified_labelarray


def joint_reassign_label_values(labelarray: np.ndarray, segments: dict,
                                name_to_label_mapping: dict) -> Tuple[np.ndarray, dict]:
    """
    Jointly reassign label values in dictionary metadata and numerical array data following
    the scheme detailed in the `name_to_label_mapping`. 
    """
    oldnew_pairs = get_old_new_label_pairs(segments, name_to_label_mapping).values()
    labelarray = modify_label_array_values(labelarray, oldnew_pairs)
    segments = reassign_label_segments_values(segments, name_to_label_mapping)
    return (labelarray, segments)


def nullify_labels(labelarray: np.ndarray, nullable_label_values: Sequence[int]) -> np.array:
    """
    Modify a label array by setting the voxels with nullable label values to zero (background)
    """
    for labelvalue in nullable_label_values:
        labelarray = np.where(labelarray == labelvalue, 0, labelarray)
    return labelarray


def sift_lowest_layer(segments: dict) -> dict:
    """Delete all non-lowest layer segments from the dictionary."""
    if not is_multilayer_segments_dict(segments):
        return segments
    # Select only the lowest layer via layer number 0. 
    return group_by_layer(segments).pop(0)


def sift_lowest_layer_labelarray(labelarray: np.ndarray) -> np.ndarray:
    """Extract lowest layer of the multilayer labelarray"""
    if labelarray.ndim == 3:
        return labelarray
    elif labelarray.ndim == 4:
        return labelarray[0, ...]
    else:
        message = (f'attempting to extract lowest layer from '
                   f'labelarray with ndim {labelarray.ndim}!'
                   f'Labelarray must have ndim 3 or 4')
        raise ValueError(message)


def compute_3D_extent(volume: np.ndarray) -> np.ndarray:
    """
    Compute the nonzero extent of an 3D volumetric array.
    The extent is specified as a 6-element array of integers:
    [I_start, I_stop, J_start, J_stop, K_start, K_stop]
    """
    indexarr = np.argwhere(volume)
    (i_start, j_start, k_start) = indexarr.min(axis=0)
    (i_stop, j_stop, k_stop) = indexarr.max(axis=0) + 1
    return np.array([i_start, i_stop, j_start, j_stop, k_start, k_stop])


def reassign_extent(segments: dict, extent: Sequence[int]) -> None:
    """
    In-place reassign the extent value in all segments of the dictionary.
    """
    extent_string = ' '.join((str(i) for i in extent))
    for segment in segments.values():
        segment['Extent'] = extent_string
    return None


def rollback_metadata_format(segments: dict) -> dict:
    """
    Roll back segments dictionary format by prepending the `SegmentN_`
    prefix and flattening the dictionary. 
    """
    flat_metadata = {}
    # Start with 1 to reserve 0 for background.
    for i, attrdict in enumerate(segments.values(), start=1):
        for key, value in attrdict.items():
            extended_key = ''.join((f'Segment{i}_', key))
            flat_metadata[extended_key] = value
    return flat_metadata


class Sifter:
    """
    Sift for specific segment names in metadata and reassign label
    values in both metadata and numerical array data.
    """

    def __init__(self,
                 names_label_map: Dict[str, int],
                 delete_multilayer: bool = True,
                 recompute_extent: bool = True) -> None:
        self.names_label_map = names_label_map
        self.delete_multilayer = delete_multilayer
        self.recompute_extent = recompute_extent

    
    @classmethod
    def from_sieberdefaults(cls) -> 'Sifter':
        """Autoconfigured specifically for Sieber et al. datasets."""
        names_label_map = {'Scala Vestibuli' : 1, 'Scala Tympani' : 2}
        return cls(names_label_map=names_label_map, delete_multilayer=True)
    

    def sift(self, labelarray: np.ndarray,
             metadata: dict) -> Tuple[np.ndarray]:
        segments, generals = parse_header(metadata)
        if self.delete_multilayer:
            segments = sift_lowest_layer(segments)
            labelarray = sift_lowest_layer_labelarray(labelarray)
            generals['sizes'] = labelarray.shape
        # Validates uniqueness of label values, ID strings and names.
        is_valid, message = is_valid_segments_dict(segments)
        if not is_valid:
            raise RuntimeError(message)

        selected, deselected = select_by_names(segments, 
                                               self.names_label_map.keys())
        labelarray = nullify_labels(labelarray, get_unique_label_values(deselected))
        # Modify label values in selected metadata and in array data.
        labelarray, selected = joint_reassign_label_values(labelarray, selected,
                                                           self.names_label_map)
        if self.recompute_extent:
            extent = compute_3D_extent(labelarray)
            reassign_extent(selected, extent)

        return (labelarray, selected, generals) 



    