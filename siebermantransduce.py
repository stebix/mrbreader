"""
Manual transduction script for the Sieber et al. datasets
"""
import json
import datetime
from pathlib import Path

from tqdm import tqdm

from reader.mrbfile import MRBFile
from reader.sieberexport import ExporterSBR
from reader.sifter import Sifter
from reader.rescaler import DummyRescaler, Rescaler


# Settings are changed here

RAW_SELECTOR = 'unembedded'
RAW_INTERPOLATION_ORDER = 3
LABEL_INTERPOLATION_ORDER = 1
CROP_TO_ROI = True
ROI_PAD_WIDTH = 30
midfix = 'E' if RAW_SELECTOR == 'embedded' else 'UE'


def create_target_fname(path: Path, int_midfix=None):
    baseID = path.name.split('-')[0]
    if int_midfix is not None:
        full_midfix = '_'.join((str(int_midfix), midfix))
    else:
        full_midfix = midfix
    # prepend a single underscore
    full_midfix = ''.join(('_', full_midfix))
    return ''.join((baseID, full_midfix, '.hdf5'))


def create_exportinfo_dict():
    fmt = '%Y-%m-%d_%H:%M:%S'
    timestamp = datetime.datetime.now().strftime(fmt)
    exportinfo = {
        'timestamp' : timestamp,
        'raw_selector' : RAW_SELECTOR,
        'raw_interpolation_order' : RAW_INTERPOLATION_ORDER,
        'label_interpolation_order' : LABEL_INTERPOLATION_ORDER,
        'crop_to_ROI' : CROP_TO_ROI,
        'ROI_pad_width' : ROI_PAD_WIDTH
    }
    return exportinfo


def main():

    # set directories here
    sieber_source_dir = Path('G:/Cochlea/dataset_sieber/landmarked')
    sieber_target_dir = Path('G:/Cochlea/dataset_sieber/transduced')


    rescaler = Rescaler(original_voxel_size=0.125,
                        rescaled_voxel_size=0.099,
                        raw_interpolation_order=RAW_INTERPOLATION_ORDER,
                        label_interpolation_order=LABEL_INTERPOLATION_ORDER,
                        device='gpu')

    # rescaler = DummyRescaler()
    sifter = Sifter.from_sieberdefaults()

    exporter = ExporterSBR(
        raw_selector=RAW_SELECTOR,
        sifter=sifter,
        rescaler=rescaler,
        crop_to_ROI=CROP_TO_ROI,
        ROI_pad_width=ROI_PAD_WIDTH
    )

    
    clean_sieber_fpaths = [
        sieber_source_dir / 'alpha-landmarked-modfin.mrb',
        sieber_source_dir / 'gamma-landmarked-bimodal.mrb',
        sieber_source_dir / 'epsilon-landmarked-bimodal.mrb',
        sieber_source_dir / 'delta-landmarked-modfin.mrb',
        sieber_source_dir / 'eta-landmarked-bimodal.mrb',
        sieber_source_dir / 'zeta-landmarked-bimodal.mrb',
        sieber_source_dir / 'theta-landmarked-bimodal.mrb'
    ]

    for idx, fpath in enumerate(tqdm(clean_sieber_fpaths), start=1):
        fname = create_target_fname(fpath, idx)
    
        mrbfile = MRBFile(fpath)
        writepath = sieber_target_dir / fname
        exporter.export(mrbfile, writepath)
    
    exportinfo_fpath = sieber_target_dir / 'exportinfo.json'
    with open(exportinfo_fpath, mode='w') as handle:
        json.dump(create_exportinfo_dict(), handle, indent=4)

if __name__ == '__main__':
    main()