import pathlib

import pytest


from reader.mrbfile import MRBFile

@pytest.fixture
def mrbfile_local():
    mrbpath = pathlib.Path(
        'G:/Cochlea/Manual_Segmentations/landmarked/8_lm.mrb'
    )
    assert mrbpath.is_file(), 'WTF'

    return MRBFile(mrbpath)


class Test:

    def test_file_exists(self, mrbfile_local):
        pass