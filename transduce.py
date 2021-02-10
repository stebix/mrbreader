import argparse
import pathlib
import itertools
from typing import Union, List, Tuple

from reader.mrbfile import MRBFile
from reader.export import HDF5Exporter

def _flatten(l, container_types=(List, Tuple)):
    for elem in l:
        if isinstance(elem, container_types):
            yield from _flatten(elem)
        else:
            yield elem


def _gather_files(candidate_paths: List[Union[str, pathlib.Path]]) -> List[pathlib.Path]:
    """
    Gather the list of HDF5 files from a heterogenous candidate path list
    that may contain directories and filepaths.
    """
    suffixes = ['.MRB', '.mrb']
    # wrap into list if candidate path is only one object
    if isinstance(candidate_paths, (str, pathlib.Path)):
        candidate_paths = [candidate_paths]

    hdf5_filepaths = []
    for c_path in candidate_paths:
        if not isinstance(c_path, pathlib.Path):
            c_path = pathlib.Path(c_path)

        if c_path.is_file():
            if c_path.suffix in suffixes:
                hdf5_filepaths.append(c_path.resolve())
        elif c_path.is_dir():
            hdf5_filepaths.append(
                _gather_files([p for p in c_path.iterdir()])
            )
    return list(_flatten(hdf5_filepaths))



def main():

    parser = argparse.ArgumentParser(
        description='Transduce 3DSlicer-produced MRB files into HDF5 DNN training data.'
    )
    parser.add_argument(
        '--source', nargs='+', type=str, required=True,
        help=('MRB source files. (Sub-) Directories are crawled recursively '
              'and files are added automatically based on suffix matching.')
    )
    parser.add_argument(
        '--target_dir', type=str, required=True,
        help='Target directory where the produced HDF5 files are stored.'
    )
    args = parser.parse_args()

    target_dir = pathlib.Path(args.target_dir)
    source_paths = _gather_files(args.source)

    if not target_dir.is_dir():
        assert not target_dir.is_file(), f'Target path < {target_dir.resolve()} > is a file!'
        target_dir.mkdir(parents=True)
    
    exporter = HDF5Exporter(raw_internal_path='raw/raw-0',
                            label_internal_path='label/label-0',
                            force_write=True)
    
    for source_file in source_paths:
        prefix = ''
        suffix = '.hdf5'
        hdf_fname = ''.join((prefix, source_file.stem, suffix))

        mrbfile = MRBFile(source_file)

        exporter.store(
            save_path=target_dir / hdf_fname,
            tagged_raw_data=mrbfile.read_raws()[0],
            tagged_label_data=mrbfile.read_segmentations()[0]
        )



if __name__ == '__main__':
    main()