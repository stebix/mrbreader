import numpy as np

from pathlib import Path

import pytest

from reader.mrbfile import MRBFile
from reader.sieberexport import ExporterSBR
from reader.sifter import Sifter
from reader.rescaler import Rescaler

def print_types(dictionary):
    for key, value in dictionary.items():
        type_of_value = type(value)
        if isinstance(value, np.ndarray):
            addinfo = f' dtype -> {value.dtype}'
        else:
            addinfo = ''
        print(f'Key :: {key} -> {type_of_value}' + addinfo)


def test_full_integration():
    raw_selector = 'unembedded'
    rescaler = Rescaler(original_voxel_size=125,
                        rescaled_voxel_size=99,
                        interpolation_order=3,
                        device='gpu')
    sifter = Sifter.from_sieberdefaults()

    exporter = ExporterSBR(
        raw_selector=raw_selector,
        sifter=sifter,
        rescaler=rescaler
    )

    fpath = Path(
        'G:/Cochlea/dataset_sieber/landmarked/zeta-landmarked-bimodal.mrb'
    )

    mrbfile = MRBFile(fpath)
    # print(mrbfile.get_member_info())

    # landmarks = mrbfile.read_landmarks()
    # print(landmarks)

    # landmarks = rescaler.rescale_landmarks(landmarks)
    # print('New Landmarks')
    # print(landmarks)

    writepath = Path('G:/Cochlea/dataset_sieber/export-gpu.hdf5')

    # label = exporter.get_label(mrbfile)
    # rescaled_general_metadata = exporter.rescaler.rescale_general_metadata(label.base_metadata)
    # rescaled_segment_metadata = exporter.rescaler.rescale_segment_metadata(label.metadata)
    # print_types(rescaled_general_metadata)
    # print_types(rescaled_segment_metadata)

    # raise Exception


    exporter.export(mrbfile, writepath)