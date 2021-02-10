import numpy as np
import pytest
import h5py

from reader.export import HDF5Exporter



class Test_HDF5Exporter:


    def test_store_only_raw(self, mock_tagged_raw_data, tmp_path):
        internal_path = 'raw/raw-0'
        exporter = HDF5Exporter(internal_path, store_metadata=True)
        test_file_path = tmp_path / 'store_only_raw_testfile.hdf5'

        # first store file
        exporter.store(save_path=test_file_path,
                       tagged_raw_data=mock_tagged_raw_data)

        # read file to check consistency
        recovered_metadata = {}
        with h5py.File(test_file_path, mode='r') as readfile:
            recovered_num_data = readfile[internal_path][...]
            for key, value in readfile[internal_path].attrs.items():
                recovered_metadata[key] = value

        # check consistency
        expected_metadata = mock_tagged_raw_data.metadata
        expected_num_data = mock_tagged_raw_data.data
        assert np.array_equal(recovered_num_data, mock_tagged_raw_data.data)
        for key in expected_metadata.keys():
            assert key in recovered_metadata.keys(), f'Missing key! {key} not present'
            assert recovered_metadata[key] == expected_metadata[key], f'Value mismatch for {key}'


    def test_store_raw_and_label(self, tmp_path,
                                 mock_tagged_raw_data, synthetic_segmentation):
        # general setup
        raw_internal_path = 'raw/raw-0'
        label_internal_path = 'label/label-0'

        exporter = HDF5Exporter(raw_internal_path=raw_internal_path,
                                label_internal_path=label_internal_path,
                                store_metadata=True)
        test_file_path = tmp_path / 'store_raw_and_label_testfile.hdf5'

        def is_equal(candidate, expected):
            """Small utility function to compare heterogenous dict values."""
            if isinstance(candidate, str):
                return candidate == expected
            else:
                # now we assume it is some sort of numerical datatype
                candidate = np.array(candidate)
                expected = np.array(expected)
                return np.allclose(candidate, expected)

        # first store file
        exporter.store(save_path=test_file_path,
                       tagged_raw_data=mock_tagged_raw_data,
                       tagged_label_data=synthetic_segmentation)
        
        # extract and compare
        raw_label_components = zip(
            [raw_internal_path, label_internal_path],
            [mock_tagged_raw_data, synthetic_segmentation]
        )
        for internal_path, tagged_data_instance in raw_label_components:
            expected_array_data = tagged_data_instance.data
            expected_metadata = tagged_data_instance.metadata
            recovered_metadata = {}

            with h5py.File(test_file_path, mode='r') as rf:
                recovered_array_data = rf[internal_path][...]
                for key, value in rf[internal_path].attrs.items():
                    recovered_metadata[key] = value

            assert_msg = f'Numerical data mismatch for object: {tagged_data_instance}'
            assert np.array_equal(recovered_array_data, expected_array_data), assert_msg
            for key in expected_metadata.keys():
                assert key in recovered_metadata, f'Missing expected key {key} in reco metadata!'
                assert is_equal(recovered_metadata[key], expected_metadata[key]), f'Metadata value mismatch for key: {key}'
            
    
    def test_overwrite_safety(self,mock_tagged_raw_data, tmp_path):
        test_file_path = tmp_path / 'overwrite_safety_mock.hdf5'
        # create a pre-existing file
        with open(test_file_path, mode='w') as writefile:
            writefile.write('Please do not overwrite me :D !!!')
        
        internal_path = 'raw/raw-0'
        exporter = HDF5Exporter(internal_path,
                                store_metadata=True,
                                force_write=False)

        # try to store file to pre-existing file path
        with pytest.raises(FileExistsError):
            exporter.store(save_path=test_file_path,
                           tagged_raw_data=mock_tagged_raw_data)



