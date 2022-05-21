import numpy as np

from pathlib import Path

import pytest

from reader.mrbfile import MRBFile
from reader.sieberexport import ExporterSBR
from reader.sifter import Sifter
from reader.rescaler import DummyRescaler, Rescaler

def print_types(dictionary):
    for key, value in dictionary.items():
        type_of_value = type(value)
        if isinstance(value, np.ndarray):
            addinfo = f' dtype -> {value.dtype}'
        else:
            addinfo = ''
        print(f'Key :: {key} -> {type_of_value}' + addinfo)


def test_full_integration():
    raw_selector = 'embedded'
    rescaler = Rescaler(original_voxel_size=0.125,
                        rescaled_voxel_size=0.099,
                        interpolation_order=2,
                        device='gpu')

    # rescaler = DummyRescaler()
    sifter = Sifter.from_sieberdefaults()

    exporter = ExporterSBR(
        raw_selector=raw_selector,
        sifter=sifter,
        rescaler=rescaler,
        crop_to_ROI=True,
        ROI_pad_width=30
    )

    fpath = Path(
        'G:/Cochlea/dataset_sieber/landmarked/theta-landmarked-bimodal.mrb'
    )

    mrbfile = MRBFile(fpath)
    # print(mrbfile.get_member_info())

    # landmarks = mrbfile.read_landmarks()
    # print(landmarks)

    # landmarks = rescaler.rescale_landmarks(landmarks)
    # print('New Landmarks')
    # print(landmarks)

    writepath = Path('G:/Cochlea/dataset_sieber/theta-pad30.hdf5')

    # label = exporter.get_label(mrbfile)
    # rescaled_general_metadata = exporter.rescaler.rescale_general_metadata(label.base_metadata)
    # rescaled_segment_metadata = exporter.rescaler.rescale_segment_metadata(label.metadata)
    # print_types(rescaled_general_metadata)
    # print_types(rescaled_segment_metadata)

    # raise Exception


    exporter.export(mrbfile, writepath)