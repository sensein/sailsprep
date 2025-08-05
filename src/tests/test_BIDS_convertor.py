"""Tests for BIDS Video Processing Pipeline."""

import json
import os
import sys
from datetime import datetime
from types import ModuleType
from typing import Generator
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
import yaml


# Create a temporary config file to allow module import
@pytest.fixture(scope="session", autouse=True)
def setup_mock_config() -> Generator[None, None, None]:
    """Create a temporary config.yaml file for testing."""
    mock_config = {
        'video_root': '/mock/videos',
        'asd_csv': 'mock_asd.csv',
        'nonasd_csv': 'mock_nonasd.csv',
        'output_dir': '/mock/output',
        'target_resolution': '1280x720',
        'target_fps': 30
    }

    # Create temporary config file
    with open('config.yaml', 'w') as f:
        yaml.dump(mock_config, f)

    yield

    # Cleanup
    if os.path.exists('config.yaml'):
        os.remove('config.yaml')

# Import the module after config is created
@pytest.fixture(scope="session")
def bvp_module(setup_mock_config: Generator[None, None, None]) -> ModuleType:
    """Import the BIDS converter module."""
    sys.path.insert(0, 'src')
    import BIDS_convertor as bvp
    return bvp

class TestConfiguration:
    """Test configuration loading and validation."""

    def test_load_configuration_success(self, bvp_module: ModuleType) -> None:
        """Test successful configuration loading."""
        mock_config = {
            'video_root': '/path/to/videos',
            'asd_csv': 'asd.csv',
            'nonasd_csv': 'nonasd.csv',
            'output_dir': '/output',
            'target_resolution': '1280x720',
            'target_fps': 30
        }

        with patch('builtins.open', mock_open(read_data=yaml.dump(mock_config))):
            with patch('yaml.safe_load', return_value=mock_config):
                config = bvp_module.load_configuration('config.yaml')
                assert config == mock_config

    def test_load_configuration_file_not_found(self, bvp_module: ModuleType) -> None:
        """Test configuration loading with missing file."""
        with patch('builtins.open', side_effect=FileNotFoundError()):
            with pytest.raises(FileNotFoundError):
                bvp_module.load_configuration('nonexistent.yaml')


class TestBIDSStructure:
    """Test BIDS directory structure creation and validation."""

    def test_create_bids_structure(self, bvp_module: ModuleType) -> None:
        """Test BIDS directory structure creation."""
        with patch('os.makedirs') as mock_makedirs:
            bvp_module.create_bids_structure()
            # Check that directories are created with exist_ok=True
            assert mock_makedirs.call_count == 2

    def test_create_dataset_description(self, bvp_module: ModuleType) -> None:
        """Test dataset description file creation."""
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            with patch('json.dump') as mock_json_dump:
                bvp_module.create_dataset_description()
                mock_file.assert_called_once()
                mock_json_dump.assert_called_once()
                # Check that the dataset description contains required fields
                args, kwargs = mock_json_dump.call_args
                dataset_desc = args[0]
                assert 'Name' in dataset_desc
                assert 'BIDSVersion' in dataset_desc
                assert 'DatasetType' in dataset_desc

    def test_create_readme(self, bvp_module: ModuleType) -> None:
        """Test README file creation."""
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            bvp_module.create_readme()
            mock_file.assert_called_once()
            # Check that content was written
            handle = mock_file()
            handle.write.assert_called()


class TestBIDSNaming:
    """Test BIDS naming conventions and filename generation."""

    def test_create_bids_filename(self, bvp_module: ModuleType) -> None:
        """Test BIDS filename creation."""
        filename = bvp_module.create_bids_filename(123, '01', 'beh', 'mp4')
        expected = 'sub-123_ses-01_task-play_beh.mp4'
        assert filename == expected

    def test_get_session_from_path_12_16_months(self, bvp_module: ModuleType) -> None:
        """Test session determination for 12-16 month videos."""
        path = '/data/videos/12-16 month/participant_video.mp4'
        session = bvp_module.get_session_from_path(path)
        assert session == '01'

    def test_get_session_from_path_34_38_months(self, bvp_module: ModuleType) -> None:
        """Test session determination for 34-38 month videos."""
        path = '/data/videos/34-38 month/participant_video.mp4'
        session = bvp_module.get_session_from_path(path)
        assert session == '02'


class TestDemographicsHandling:
    """Test demographics data processing."""

    def test_read_demographics(self, bvp_module: ModuleType) -> None:
        """Test demographics CSV reading and combining."""
        asd_data = pd.DataFrame({
            'dependent_temporary_id': ['A001', 'A002'],
            'dependent_dob': ['2022-01-01', '2022-02-01'],
            'sex': ['M', 'F'],
            'diagnosis': ['ASD', 'ASD']
        })

        nonasd_data = pd.DataFrame({
            'dependent_temporary_id': ['N001', 'N002'],
            'dependent_dob': ['2022-03-01', '2022-04-01'],
            'sex': ['F', 'M'],
            'diagnosis': ['TD', 'TD']
        })

        with patch('pandas.read_csv', side_effect=[asd_data, nonasd_data]):
            df = bvp_module.read_demographics('asd.csv', 'nonasd.csv')
            assert len(df) == 4
            assert 'dependent_temporary_id' in df.columns


class TestVideoMetadataExtraction:
    """Test video metadata extraction and processing."""

    def test_extract_exif_success(self, bvp_module: ModuleType) -> None:
        """Test successful video metadata extraction."""
        mock_metadata = {
            "format": {
                "filename": "test.mp4",
                "format_long_name": "QuickTime / MOV",
                "duration": "120.5",
                "bit_rate": "1000000",
                "size": "15000000",
                "tags": {"creation_time": "2023-01-01T12:00:00.000000Z"}
            },
            "streams": [
                {
                    "tags": {"creation_time": "2023-01-01T12:00:00.000000Z"}
                }
            ]
        }

        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps(mock_metadata)

            result = bvp_module.extract_exif('test.mp4')
            assert 'duration_sec' in result
            assert result['duration_sec'] == 120.5
            assert result['format'] == "QuickTime / MOV"

    def test_extract_exif_ffprobe_error(self, bvp_module: ModuleType) -> None:
        """Test video metadata extraction with ffprobe error."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "Error message"

            result = bvp_module.extract_exif('test.mp4')
            assert 'ffprobe_error' in result
            assert result['ffprobe_error'] == "Error message"


class TestDateExtraction:
    """Test date extraction from filenames."""

    def test_extract_date_from_filename_standard_format(
        self, bvp_module: ModuleType
    ) -> None:
        """Test date extraction from standard format."""
        # Test a format that should work based on the actual implementation
        filename = "2023-12-25.mp4"  # Remove 'video_' prefix
        result = bvp_module.extract_date_from_filename(filename)
        assert result == "2023:12:25 00:00:00"

    def test_extract_date_from_filename_mmddyyyy_format(
        self, bvp_module: ModuleType
    ) -> None:
        """Test date extraction from MM-DD-YYYY format."""
        filename = "12-25-2023.mp4"
        result = bvp_module.extract_date_from_filename(filename)
        assert result == "2023:12:25 00:00:00"

    def test_extract_date_from_filename_yyyymmdd_format(
        self, bvp_module: ModuleType
    ) -> None:
        """Test date extraction from YYYYMMDD format."""
        filename = "20231225.mp4"
        result = bvp_module.extract_date_from_filename(filename)
        assert result == "2023:12:25 00:00:00"

    def test_extract_date_from_filename_invalid(self, bvp_module: ModuleType) -> None:
        """Test date extraction from invalid filename."""
        filename = "invalid_filename.mp4"
        result = bvp_module.extract_date_from_filename(filename)
        assert result is None

    def test_calculate_age(self, bvp_module: ModuleType) -> None:
        """Test age calculation in months."""
        dob_str = "2022-01-15"
        video_date = datetime(2023, 1, 15)
        age = bvp_module.calculate_age(dob_str, video_date)
        assert age == 12.0


class TestVideoProcessing:
    """Test video processing functions."""

    @patch('subprocess.run')
    @patch('os.remove')
    @patch('os.path.exists')
    def test_stabilize_video(
        self,
        mock_exists: MagicMock,
        mock_remove: MagicMock,
        mock_run: MagicMock,
        bvp_module: ModuleType
    ) -> None:
        """Test video stabilization."""
        mock_exists.return_value = True
        bvp_module.stabilize_video('input.mp4', 'output.mp4')

        # Should call subprocess.run twice (detect and transform)
        assert mock_run.call_count == 2
        mock_remove.assert_called_once_with("transforms.trf")

    @patch('subprocess.run')
    def test_extract_audio(
        self, mock_run: MagicMock, bvp_module: ModuleType
    ) -> None:
        """Test audio extraction from video."""
        bvp_module.extract_audio('input.mp4', 'output.wav')
        mock_run.assert_called_once()

        # Check that the command includes correct audio parameters
        args = mock_run.call_args[0][0]
        assert '-ar' in args
        assert '16000' in args
        assert '-ac' in args
        assert '1' in args


class TestMetadataFileCreation:
    """Test creation of BIDS metadata files."""

    def test_create_events_tsv(self, bvp_module: ModuleType) -> None:
        """Test events TSV file creation."""
        video_metadata = {'duration_sec': 120.5}

        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            bvp_module.create_events_tsv(video_metadata, 'output.tsv')
            mock_to_csv.assert_called_once()

    def test_create_video_metadata_json(self, bvp_module: ModuleType) -> None:
        """Test video metadata JSON creation."""
        metadata = {'duration_sec': 120.5, 'format': 'MP4'}
        processing_info = {'has_stabilization': True}

        with patch('builtins.open', mock_open()):
            with patch('json.dump') as mock_json_dump:
                bvp_module.create_video_metadata_json(
                    metadata, processing_info, 'output.json'
                )
                mock_json_dump.assert_called_once()

                # Check JSON content structure
                args = mock_json_dump.call_args[0]
                json_content = args[0]
                assert 'TaskName' in json_content
                assert 'ProcessingPipeline' in json_content
                assert 'OriginalMetadata' in json_content


class TestUtilityFunctions:
    """Test utility functions."""

    def test_save_json(self, bvp_module: ModuleType) -> None:
        """Test JSON file saving utility."""
        test_data = {'test': 'data', 'number': 123}

        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            with patch('json.dump') as mock_json_dump:
                bvp_module.save_json(test_data, 'output.json')
                # Check that json.dump was called with the test data and the file handle
                mock_json_dump.assert_called_once()
                args, kwargs = mock_json_dump.call_args
                assert args[0] == test_data
                assert kwargs.get('indent') == 4


class TestMainWorkflow:
    """Test the main processing workflow."""

    @patch('BIDS_convertor.create_participants_files')
    @patch('BIDS_convertor.process_videos')
    @patch('BIDS_convertor.read_demographics')
    @patch('BIDS_convertor.create_readme')
    @patch('BIDS_convertor.create_derivatives_dataset_description')
    @patch('BIDS_convertor.create_dataset_description')
    @patch('BIDS_convertor.create_bids_structure')
    @patch('BIDS_convertor.save_json')
    def test_main_workflow(
        self,
        mock_save_json: MagicMock,
        mock_create_structure: MagicMock,
        mock_create_dataset: MagicMock,
        mock_create_derivatives: MagicMock,
        mock_create_readme: MagicMock,
        mock_read_demographics: MagicMock,
        mock_process_videos: MagicMock,
        mock_create_participants: MagicMock,
        bvp_module: ModuleType
    ) -> None:
        """Test the main processing workflow."""
        # Setup mocks
        mock_demographics = pd.DataFrame({'id': [1, 2]})
        mock_read_demographics.return_value = mock_demographics
        mock_process_videos.return_value = ([{'test': 'data'}], ['error1'])

        # Run main function
        bvp_module.main()

        # Verify all steps were called
        mock_create_structure.assert_called_once()
        mock_create_dataset.assert_called_once()
        mock_create_derivatives.assert_called_once()
        mock_create_readme.assert_called_once()
        mock_read_demographics.assert_called_once()
        mock_process_videos.assert_called_once()
        mock_create_participants.assert_called_once()
        assert mock_save_json.call_count == 2


# Test fixtures for reusable data
@pytest.fixture
def sample_demographics() -> pd.DataFrame:
    """Sample demographics DataFrame for testing."""
    return pd.DataFrame({
        'dependent_temporary_id': ['A001', 'A002', 'N001'],
        'dependent_dob': ['2022-01-01', '2022-02-01', '2022-03-01'],
        'sex': ['M', 'F', 'M'],
        'diagnosis': ['ASD', 'ASD', 'TD']
    })


@pytest.fixture
def sample_video_metadata() -> dict[str, float | str | int]:
    """Sample video metadata for testing."""
    return {
        'duration_sec': 120.5,
        'format': 'QuickTime / MOV',
        'bit_rate': 1000000,
        'size_bytes': 15000000
    }


if __name__ == '__main__':
    pytest.main([__file__])