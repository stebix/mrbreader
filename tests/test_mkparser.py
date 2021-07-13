import collections.abc
import pytest

from reader.mkparser import (_construct_fiducial_markup_dict,
                             extract_fiducial_markups)


@pytest.fixture
def mock_fiducial_dict():
    """
    Copied test/mock fiducial information dictionary for testing purposes.
    Taken directly from zipfile-loaded JSON element of the MRB file.
    """
    d = {'@schema': 'https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#',
         'markups': [{'type': 'Fiducial',
         'coordinateSystem': 'LPS',
         'locked': False,
         'labelFormat': '%N-%d',
         'controlPoints': [{'id': '1',
             'label': 'CochleaTop',
             'description': '',
             'associatedNodeID': 'vtkMRMLScalarVolumeNode1',
             'position': [-45.203976811025875,
             -169.9861093800334,
             -152.83510139375863],
             'orientation': [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
             'selected': True,
             'locked': False,
             'visibility': True,
             'positionStatus': 'defined'}],
         'measurements': [],
         'display': {'visibility': True,
             'opacity': 1.0,
             'color': [0.4, 1.0, 0.0],
             'selectedColor': [1.0, 0.5000076295109483, 0.5000076295109483],
             'activeColor': [0.4, 1.0, 0.0],
             'propertiesLabelVisibility': False,
             'pointLabelsVisibility': True,
             'textScale': 3.0,
             'glyphType': 'Sphere3D',
             'glyphScale': 1.0,
             'glyphSize': 5.0,
             'useGlyphScale': True,
             'sliceProjection': False,
             'sliceProjectionUseFiducialColor': True,
             'sliceProjectionOutlinedBehindSlicePlane': False,
             'sliceProjectionColor': [1.0, 1.0, 1.0],
             'sliceProjectionOpacity': 0.6,
             'lineThickness': 0.2,
             'lineColorFadingStart': 1.0,
             'lineColorFadingEnd': 10.0,
             'lineColorFadingSaturation': 1.0,
             'lineColorFadingHueOffset': 0.0,
             'handlesInteractive': False,
             'snapMode': 'toVisibleSurface'}}]}
    return d


class Test_extract_fiducial_markups:

    def test_single_extract(self, mock_fiducial_dict):
        fid_markups_list = extract_fiducial_markups(mock_fiducial_dict)
        assert isinstance(fid_markups_list, collections.abc.Sequence), 'Expecting list of dict'
        assert len(fid_markups_list) == 1, 'Expecting singleton list'
        print(fid_markups_list)

    