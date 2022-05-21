

from reader.roi import ROISpec, Roifier
from reader.utils import extent_string_as_array


class Test_ROISpec:

    def test(self):

        extent = '10 20 100 150 200 225'
        oshape = (500, 300, 400)
        pad_width = 50

        roispec = ROISpec.from_extent(extent, pad_width, oshape)

        roifier = Roifier(roispec)

        extent = extent_string_as_array(extent)
        
        print(roifier.roify_extent(extent, 'string'))
        print(roispec.shape)