import logging

from pathlib import Path, PurePath
from typing import List

from reader.mrbfile import MRBFile
from reader.tagged_data import RawData


class ExporterSBR:
    """
    Exports the Sieber et al. styled MRB datasets to HDF5.
    """
    valid_raw_policy = ('embedded', 'unembedded', 'joint', 'separate')

    def __init__(self,
                 raw_policy: str,
                 force_write: bool = False,
                 store_metadata: bool = True) -> None:
        
        if raw_policy not in self.valid_raw_policy:
            raise ValueError(
                f'Raw data saving policy for Sieber et al. datasets '
                f'must be one of {self.valid_raw_policy}, got {raw_policy} instead'
            )
        self.raw_policy = raw_policy
        self.force_write = force_write
        self.store_metadata = store_metadata
        self._hdf5_write_mode = 'w' if force_write else 'x'
        self.logger = logging.getLogger(__name__)
    

    def check_write_location(self, savepath: Path) -> None:
        """
        Check write location and raise exception depending
        on desired overwrite behaviour.
        """
        if savepath.is_file() and not self.force_write:
            message = f'File already existing at location < "{savepath.resolve()}" >'
            raise FileExistsError(message)
        if savepath.is_dir():
            message = f'Indicated location < "{savepath.resolve()}" > is a directory'
            raise IsADirectoryError(message)


    def export(self, mrbfile: MRBFile, savepath: Path) -> None:
        self.check_write_location(savepath)
        

    def get_embedded_raw_data(self, mrbfile: MRBFile) -> RawData:
        try:
            embedded_raw = self.get_matching_raw_data(
                'CBCT_EMBEDDED', mrbfile
            )
        except FileNotFoundError:
            self.logger.warning(
                f'No embedded raw dataset present for MRBFile at {mrbfile.filepath.resolve()}'
            )
            return None
        return embedded_raw

    
    def get_unembedded_raw_data(self, mrbfile: MRBFile) -> RawData:
        try:
            unembedded_raw = self.get_matching_raw_data(
                'CBCT_UNEMBEDDED', mrbfile
            )
        except FileNotFoundError:
            self.logger.warning(
                f'No unembedded raw dataset present for MRBFile at {mrbfile.filepath.resolve()}'
            )
            return None
        return unembedded_raw
        
    
    @staticmethod
    def get_matching_raw_data(matchstring: str, mrbfile: MRBFile) -> RawData:
        """
        Tailored to Sieber dataset: Get the embedded hi-res dataset by analyzing
        ZipInfo attributes.
        """
        matching_raw = mrbfile.read_stringmatched_raws(matchstring)
        if len(matching_raw) == 0:
            raise FileNotFoundError(
                f'No embedded raw data file found in {mrbfile}'
            )
        if len(matching_raw) > 1:
            raise RuntimeError(
                f'Invalid MRBFile state: Found multiple embedded datasets'
            )
        return matching_raw[0]
    
    @staticmethod
    def generate_final_save_path(save_policy: str) -> List[Path]:
        """
        Generate the final saving path depending on the indicated save policy.
        The save policy determines whether embedded and unembedded datasets are
        stored jointly or separately.
        """
        pass


    
    @staticmethod
    def generate_generic_ipath(stem: str, i: int) -> PurePath:
        """Generate generic internal path for HDF5 files."""
        return PurePath(stem, f'{stem}-{i}')

    def generate_raw_ipath(self, i: int) -> PurePath:
        """HDF5 internal path for raw data."""
        return self.generate_generic_ipath('raw', i)

    def generate_label_ipath(self, i: int) -> PurePath:
        """HDF5 internal path for raw data."""
        return self.generate_generic_ipath('label', i)