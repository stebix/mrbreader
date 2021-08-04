"""
Parse dictionary objects derived from MRB-file JSON data
"""

from typing import Dict, List, Tuple, Any, Optional

# provides the mapping from 3DSlicer internal keywords to our preferred
# keywords for fiducial markuo attributes
KEYMAPPING = {
        'coordinateSystem' : 'coordsys',
        'id' : 'id',
        'label' : 'label',
        'position' : 'xyz_position',
        'orientation' : 'orientation'
}

# fiducial markup templates
cochlea_top_equivalents = frozenset(
    ('CochleaTop', 'cochleatop')
)
oval_window_equivalents = frozenset(
    ('OvalWindow', 'ovalwindow')
)
round_window_equivalents = frozenset(
    ('RoundWindow', 'roundwindow')
)

MARKUP_TEMPLATE = {
    cochlea_top_equivalents : {
        'label' : 'CochleaTop',
        'id' : 1
    },
    oval_window_equivalents : {
        'label' : 'OvalWindow',
        'id' : 2
    },
    round_window_equivalents : {
        'label' : 'RoundWindow',
        'id' : 3
    }
}


def _construct_fiducial_markup_dict(fid_dict: Dict, keymapping: Dict = KEYMAPPING) -> Dict:
    """
    Perform a dict-2-dict transform of relevant information from the
    raw fiducial markup data dict. 
    """
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
    return {new_key : fid_dict[old_key] for old_key, new_key in keymapping.items()}


def fit_to_template(fid_markup_dict: Dict, template: Dict = MARKUP_TEMPLATE) -> Dict:
    """
    In-place transform the given fiducial markup dictionary `fig_markup_dict`
    to match the layout from the template dictionary.
    """
    # clean the label, is sometines appended `-{N}` by the 3DSlicer
    label = fid_markup_dict['label'].split('-')[0]
    for keyset, subtemplate in template.items():
        if label in keyset:
            fid_markup_dict.update(subtemplate)
    return fid_markup_dict



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

