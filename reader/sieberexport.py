import logging

from pathlib import Path, PurePosixPath
from typing import Dict, List, Sequence

import h5py

from reader.mrbfile import MRBFile
from reader.rescaler import Rescaler
from reader.sifter import Sifter, rollback_metadata_format
from reader.tagged_data import LabelData, RawData
from reader.hdf5tools import (create_dataset_from_path,
                              create_groups_from_path,
                              generate_internal_path,
                              write_to_attrs, bulk_write_to_attrs)


class ExporterSBR:
    """
    Exports the Sieber et al. styled MRB datasets to HDF5.
    """
    valid_raw_selectors = ('embedded', 'unembedded')

    def __init__(self,
                 raw_selector: str,
                 sifter: Sifter,
                 rescaler: Rescaler,
                 force_write: bool = False,
                 store_metadata: bool = True) -> None:
        
        if raw_selector not in self.valid_raw_selectors:
            raise ValueError(
                f'Raw data selector for Sieber et al. datasets '
                f'must be one of {self.valid_raw_selectors}, got {raw_selector} instead'
            )
        self.rescaler = rescaler
        self.sifter = sifter
        # Saving settings
        self.raw_policy = raw_selector
        self.force_write = force_write
        self.store_metadata = store_metadata
        # Internal stuff
        self._hdf5_write_mode = 'w' if force_write else 'x'
        self._logger = logging.getLogger(__name__)
    

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

    
    def get_raw(self, mrbfile: MRBFile):
        if self.raw_policy == 'embedded':
            raw = self.get_matching_raw_data('CBCT_Embedded', mrbfile)
        elif self.raw_policy == 'unembedded':
            raw = self.get_matching_raw_data('CBCT_Unembedded', mrbfile)
        else:
            raise RuntimeError('Hic sunt dracones')
        return raw

    
    def get_label(self, mrbfile: MRBFile):
        label = mrbfile._read_segmentations()
        if len(label) > 1:
            raise RuntimeError(f'retrieved {len(label)} label arrays, expected only one')
        # Squeeze list and split the single (array, header) tuple.
        labelarray, labelheader, generals = self.sifter.sift(label[0][0], label[0][1])
        # During the processing pipeline, we excised the `SegmentN_` prefix.
        # LabelData however relies on this during its parsing operation, so we have
        # to put it back in.
        labelheader = rollback_metadata_format(labelheader)
        labelheader.update(generals)
        return LabelData(labelarray, labelheader)

    
    def get_landmarks(self, mrbfile: MRBFile):
        return mrbfile.read_landmarks()


    def export(self, mrbfile: MRBFile, savepath: Path) -> None:
        # Check validity of write location, raises exceptions if inappropriate.
        self.check_write_location(savepath)
        self._logger.info('Reading raw data')
        raw = self.get_raw(mrbfile)
        self._logger.info('Reading label data')
        label = self.get_label(mrbfile)
        self._logger.info('Reading landmark data')
        landmarks = self.get_landmarks(mrbfile)

        with h5py.File(savepath, mode=self._hdf5_write_mode) as handle:
            self._export_raw(raw, handle)
            self._export_label(label, handle)
            self._export_landmarks(landmarks, handle)
        
    
    def _export_raw(self, raw: RawData, handle: h5py.File) -> None:
        # Adjust voxel size in numerical data and metadata.
        rescaled_raw = self.rescaler.rescale_volume(raw.data)
        rescaled_metadata = self.rescaler.rescale_general_metadata(raw.metadata)
        # Perform comparison sanity check.
        if not rescaled_raw.shape == tuple(rescaled_metadata['sizes']):
            message = (f'actual and expected rescaled raw volume shape mismatch: '
                       f'actual :: {rescaled_raw.shape} vs. expected :: {tuple(rescaled_metadata["sizes"])} '
                        ' -> please investigate')
            raise RuntimeError(message)
        # Write rescaled numpy.ndarray to HDF5 file
        rawpath = generate_internal_path(stem='raw', i=0)
        dataset = create_dataset_from_path(rawpath, handle, rescaled_raw)
        if self.store_metadata:
            original_metadata_group = create_groups_from_path(
                PurePosixPath('original_metadata', 'raw'), handle
            )
            write_to_attrs(dataset, rescaled_metadata)
            write_to_attrs(original_metadata_group, raw.metadata)


    def _export_label(self, label: LabelData, handle: h5py.File) -> None:
        """Separate export method: Stores both metadata and base metadata"""
        # Adjust voxel size in numerical data and metadata.
        rescaled_label = self.rescaler.rescale_volume(label.data)
        rescaled_general_metadata = self.rescaler.rescale_general_metadata(label.base_metadata)
        rescaled_segment_metadata = self.rescaler.rescale_segment_metadata(label.metadata)
        # Write rescaled numpy.ndarray to HDF5 file
        labelpath = generate_internal_path(stem='label', i=0)
        dataset = create_dataset_from_path(labelpath, handle, rescaled_label)
        if self.store_metadata:
            # Store rescaled metadata directly as dataset attributes
            bulk_write_to_attrs(dataset, (rescaled_general_metadata, rescaled_segment_metadata))
            # Original metadata gets a separate group.
            original_metadata_group = create_groups_from_path(
                PurePosixPath('original_metadata', 'label'), handle
            )
            bulk_write_to_attrs(original_metadata_group, (label.base_metadata, label.metadata))


    def _export_landmarks(self, landmarks: Sequence[Dict], handle: h5py.File) -> None:
        rescaled_landmarks = self.rescaler.rescale_landmarks(landmarks)
        for i, landmark in enumerate(rescaled_landmarks):
            internal_path = generate_internal_path(stem='landmark', i=i)
            dataset = create_dataset_from_path(internal_path, handle, data=None)
            write_to_attrs(dataset, landmark)

    
    @staticmethod
    def get_matching_raw_data(matchstring: str, mrbfile: MRBFile) -> RawData:
        """
        Tailored to Sieber dataset: Get the embedded hi-res dataset by analyzing
        ZipInfo attributes.
        """
        matching_raw = mrbfile.read_stringmatched_raws(matchstring)
        if len(matching_raw) == 0:
            raise FileNotFoundError(
                f'No raw data file found in {mrbfile} for matchstring "{matchstring}"'
            )
        if len(matching_raw) > 1:
            raise RuntimeError(
                f'Invalid MRBFile state: Found multiple datasets for matchstring "{matchstring}"'
            )
        return matching_raw[0]




