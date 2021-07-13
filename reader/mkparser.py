"""
Parse dictionary objects derived from MRB-file JSON data
"""

from typing import Dict, List, Tuple, Any, Optional


def _construct_fiducial_markup_dict(fid_dict: Dict) -> Dict:
    """
    Perform a dict-2-dict transform of relevant information from the
    raw fiducial markup data dict. 
    """
    kw_map = {
        'coordinateSystem' : 'coordsys',
        'id' : 'id',
        'label' : 'label',
        'position' : 'xyz_position',
        'orientation' : 'orientation'
    }
    # flatten the fiducial markup dict
    # we assume that `controlPoints` value is a singelton
    # list since fiducial markups are used to mark points
    subdict_list = fid_dict['controlPoints']
    if len(subdict_list) != 1:
        raise RuntimeError(
            (f'Markup information dictionary for type Fiducial has a control point list '
             f'of length {len(subdict_list)}. Expecting length 1 for type '
             f'fiducial!')
        )
    subdict = subdict_list[0]
    fid_dict.update(subdict)
    # produce clean instance of information dict
    return {new_key : fid_dict[old_key] for old_key, new_key in kw_map.items()}



def extract_fiducial_markups(mkdict: Dict) -> List[Dict]:
    """
    Extract the fiducial markups information from the metadata dictionary
    as a list of dictionaries.
    """
    fiducial_markups = []
    for markup in mkdict['markups']:
        fiducial_markups.append(
            _construct_fiducial_markup_dict(markup)
        )
    return fiducial_markups

