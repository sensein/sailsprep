"""BIDS Video Processing Pipeline.

This module processes home videos from ASD screening studies and organizes them
according to the Brain Imaging Data Structure (BIDS) specification version 1.8.0.

The pipeline includes video stabilization, denoising, standardization, and audio
extraction for behavioral analysis research.

Example:
    Basic usage:
        $ python bids_video_processor.py

Todo:
    * check with actual data
"""

# Standard library imports
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import pandas as pd
import yaml
from dateutil import parser


def load_configuration(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration dictionary containing video processing parameters.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If the YAML file is malformed.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Load configuration
config = load_configuration()
VIDEO_ROOT = config["video_root"]
ASD_CSV = config["asd_csv"]
NONASD_CSV = config["nonasd_csv"]
OUTPUT_DIR = config["output_dir"]
TARGET_RESOLUTION = config.get("target_resolution", "1280x720")
TARGET_FRAMERATE = config.get("target_fps", 30)

# BIDS directory structure
BIDS_ROOT = os.path.join(OUTPUT_DIR, "bids-dataset")
DERIVATIVES_DIR = os.path.join(BIDS_ROOT, "derivatives", "preprocessed")


def create_bids_structure() -> None:
    """Create the BIDS directory structure.

    Creates the main BIDS dataset directory and derivatives subdirectory
    following BIDS specification requirements.

    Note:
        This function creates directories with exist_ok=True to prevent
        errors if directories already exist.
    """
    os.makedirs(BIDS_ROOT, exist_ok=True)
    os.makedirs(DERIVATIVES_DIR, exist_ok=True)


def create_dataset_description() -> None:
    """Create dataset_description.json for main BIDS dataset.

    Generates the required dataset description file according to BIDS
    specification, containing metadata about the dataset including name,
    version, authors, and description.

    Raises:
        IOError: If unable to write the dataset description file.
    """
    dataset_desc = {
        "Name": "Home Videos",
        "BIDSVersion": "1.10.0",
        "HEDVersion": "8.2.0",
        "DatasetType": "raw",
        "License": "",
        "Authors": ["Research Team"],
        "Acknowledgements": "participants and families",
        "HowToAcknowledge": "",
        "Funding": ["", "", ""],
        "EthicsApprovals": [""],
        "ReferencesAndLinks": ["", "", ""],
        "DatasetDOI": "doi:",
    }

    with open(os.path.join(BIDS_ROOT, "dataset_description.json"), "w") as f:
        json.dump(dataset_desc, f, indent=4)


def create_derivatives_dataset_description() -> None:
    """Create dataset_description.json for derivatives.

    Generates the dataset description file for the derivatives directory,
    documenting the preprocessing pipeline and source datasets.

    Raises:
        IOError: If unable to write the derivatives dataset description file.
    """
    derivatives_desc = {
        "Name": "Home Videos",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "Video Preprocessing Pipeline",
                "Version": "1.0.0",
                "Description": (
                    "FFmpeg-based video stabilization, denoising, "
                    "and standardization pipeline"
                ),
                "CodeURL": "local",
            }
        ],
        "SourceDatasets": [{"DOI": "", "URL": "", "Version": "1.0.0"}],
        "HowToAcknowledge": "Please cite the original study",
    }

    derivatives_path = os.path.join(DERIVATIVES_DIR, "dataset_description.json")
    with open(derivatives_path, "w") as f:
        json.dump(derivatives_desc, f, indent=4)


def create_readme() -> None:
    """Create README file for the BIDS dataset.

    Generates a comprehensive README file documenting the dataset structure,
    organization, processing pipeline, and usage instructions following
    BIDS best practices.

    Raises:
        IOError: If unable to write the README file.
    """
    readme_content = """# README

This README serves as the primary guide for researchers using this BIDS-format dataset.

## Details Related to Access to the Data

### Data User Agreement

### Contact Person
- Name:
- Email:
- ORCID:

### Practical Information to Access the Data

## Overview

### Project Information
- Project Name: [If applicable]
- Years: [YYYY-YYYY]

### Dataset Description
This dataset contains [brief description of data types and sample size].

### Experimental Design


### Quality Assessment
[Summary statistics or QC metrics]

## Methods

### Subjects
[Description of participant pool]

#### Recruitment
[Recruitment procedures]

#### Inclusion Criteria
1. [Criterion 1]
2. [Criterion 2]

#### Exclusion Criteria
1. [Criterion 1]
2. [Criterion 2]

### Apparatus
[Equipment and environment details]

### Initial Setup
[Pre-session procedures]

### Task Organization
- Counterbalancing: [Yes/No]
- Session Structure:
  1. [Activity 1]
  2. [Activity 2]

### Task Details


### Additional Data Acquired


### Experimental Location
[Facility/geographic details]

### Missing Data
- Participant [ID]: [Issue description]
- Participant [ID]: [Issue description]

### Notes
[Any additional relevant information]

"""

    with open(os.path.join(BIDS_ROOT, "README"), "w") as f:
        f.write(readme_content)


def get_session_from_path(video_path: Union[str, Path]) -> str:
    """Determine session ID based on video path.

    Analyzes the video file path to determine which session (age group)
    the video belongs to based on folder naming conventions.

    Args:
        video_path (str or Path): Path to the video file.

    Returns:
        str: Session ID ('01' for 12-16 months, '02' for 34-38 months).

    Note:
        Defaults to session '01' if no clear age group indicator is found.
    """
    path_str = str(video_path).lower()
    if "12-16 month" in path_str:
        return "01"
    elif "34-38 month" in path_str:
        return "02"
    else:
        # Fallback - try to infer from folder structure
        return "01"  # Default to session 01


def create_bids_filename(
    participant_id: int, session_id: str, suffix: str, extension: str
) -> str:
    """Create BIDS-compliant filename.

    Generates standardized filenames following BIDS naming conventions
    for participant data files.

    Args:
        participant_id (int): Numeric participant identifier.
        session_id (str): Session identifier (e.g., '01', '02').
        suffix (str): File type suffix (e.g., 'beh', 'events').
        extension (str): File extension without dot (e.g., 'mp4', 'tsv').

    Returns:
        str: BIDS-compliant filename.

    Example:
        >>> create_bids_filename(123, '01', 'beh', 'mp4')
        'sub-123_ses-01_task-play_beh.mp4'
    """
    return f"sub-{participant_id:02d}_ses-{session_id}_task-play_{suffix}.{extension}"


def read_demographics(asd_csv: str, nonasd_csv: str) -> pd.DataFrame:
    """Read and combine demographics data from CSV files.

    Loads participant demographics from separate ASD and non-ASD CSV files,
    combines them, and standardizes column names.

    Args:
        asd_csv (str): Path to ASD participants CSV file.
        nonasd_csv (str): Path to non-ASD participants CSV file.

    Returns:
        pd.DataFrame: Combined demographics dataframe with standardized column names.

    Raises:
        FileNotFoundError: If either CSV file is not found.
        pd.errors.EmptyDataError: If CSV files are empty.
    """
    df_asd = pd.read_csv(asd_csv)
    df_nonasd = pd.read_csv(nonasd_csv)
    df = pd.concat([df_asd, df_nonasd], ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def create_participants_files(
    demographics_df: pd.DataFrame, processed_data: List[Dict[str, Any]]
) -> None:
    """Create participants.tsv and participants.json files.

    Generates BIDS-compliant participant information files including
    a TSV file with participant data and a JSON data dictionary.

    Args:
        demographics_df (pd.DataFrame): Demographics dataframe.
        processed_data (list): List of processed video data dictionaries.

    Raises:
        IOError: If unable to write participant files.
    """
    # Get unique participants from processed data
    processed_participants = set()
    for entry in processed_data:
        processed_participants.add(entry["bids_participant_id"])

    # Filter demographics for only processed participants
    participants_data = []
    for _, row in demographics_df.iterrows():
        participant_id = str(row["dependent_temporary_id"]).upper()
        # Create consistent numeric ID
        bids_id = f"sub-{hash(participant_id) % 10000:04d}"

        if bids_id in processed_participants:
            participants_data.append(
                {
                    "participant_id": bids_id,
                    "age": row.get("dependent_dob", "n/a"),
                    "sex": row.get("sex", "n/a"),
                    "group": (
                        "ASD"
                        if "asd" in str(row.get("diagnosis", "")).lower()
                        else "NonASD"
                    ),
                }
            )

    # Create participants.tsv
    participants_df = pd.DataFrame(participants_data)
    participants_df.to_csv(
        os.path.join(BIDS_ROOT, "participants.tsv"), sep="\t", index=False
    )

    # Create participants.json (data dictionary)
    participants_json = {
        "participant_id": {"Description": "Unique participant identifier"},
        "age": {"Description": "Date of birth", "Units": "YYYY-MM-DD"},
        "sex": {
            "Description": "Biological sex of participant",
            "Levels": {"M": "male", "F": "female"},
        },
        "group": {
            "Description": "Participant group classification",
            "Levels": {
                "ASD": "Autism Spectrum Disorder",
                "NonASD": "Not Autism Spectrum Disorder",
            },
        },
    }

    with open(os.path.join(BIDS_ROOT, "participants.json"), "w") as f:
        json.dump(participants_json, f, indent=4)


def extract_exif(video_path: str) -> Dict[str, Any]:
    """Extract video metadata using ffprobe.

    Uses FFmpeg's ffprobe tool to extract comprehensive metadata from video files
    including format information, stream details, and embedded timestamps.

    Args:
        video_path (str): Path to the video file.

    Returns:
        dict: Dictionary containing extracted metadata including duration,
              bit rate, format information, and date/time tags.

    Note:
        Returns error information in the dictionary if ffprobe fails
        or if the video format is unsupported.

    Example:
        >>> metadata = extract_exif('/path/to/video.mp4')
        >>> print(metadata['duration_sec'])
        120.5
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"ffprobe_error": result.stderr.strip()}
        metadata = json.loads(result.stdout)
        extracted = {}
        # Format-level metadata
        format_info = metadata.get("format", {})
        extracted["filename"] = format_info.get("filename")
        extracted["format"] = format_info.get("format_long_name")
        extracted["duration_sec"] = float(format_info.get("duration", 0))
        extracted["bit_rate"] = int(format_info.get("bit_rate", 0))
        extracted["size_bytes"] = int(format_info.get("size", 0))
        # Date/time-related tags from format
        extracted["format_dates"] = {}
        if "tags" in format_info:
            for k, v in format_info["tags"].items():
                if "date" in k.lower() or "time" in k.lower():
                    extracted["format_dates"][k] = v
        # Loop through all streams (video, audio, etc.)
        extracted["stream_dates"] = []
        for stream in metadata.get("streams", []):
            stream_entry = {}
            if "tags" in stream:
                for k, v in stream["tags"].items():
                    if "date" in k.lower() or "time" in k.lower():
                        stream_entry[k] = v
            if stream_entry:
                extracted["stream_dates"].append(stream_entry)
        return extracted
    except Exception as e:
        return {"error": str(e)}


def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract date from filename using various patterns.

    Attempts to parse dates from video filenames using multiple common
    date formats and patterns, including Facebook/Instagram formats
    and standard date conventions.

    Args:
        filename (str): Video filename to parse.

    Returns:
        str or None: Formatted date string in "YYYY:MM:DD HH:MM:SS" format,
                     or None if no valid date pattern is found.

    Note:
        This function tries multiple date formats and patterns to maximize
        compatibility with various naming conventions used by different
        devices and platforms.

    Example:
        >>> extract_date_from_filename('video_2023-12-25.mp4')
        '2023:12:25 00:00:00'
    """
    try:
        name = os.path.splitext(os.path.basename(filename))[0]
        # Try direct known formats
        known_formats = [
            "%m-%d-%Y",
            "%m-%d-%y",
            "%m_%d_%Y",
            "%m_%d_%y",
            "%Y-%m-%d",
            "%Y%m%d",
            "%m%d%Y",
        ]
        for fmt in known_formats:
            try:
                return datetime.strptime(name, fmt).strftime("%Y:%m:%d %H:%M:%S")
            except ValueError:
                continue
        # Try extracting from YYYYMMDD_HHMMSS or FB_/IMG_ formats
        match = re.search(r"(20\d{6})[_\-]?(?:([01]\d{3,4}))?", name)
        if match:
            date_str = match.group(1)
            time_str = match.group(2) if match.group(2) else "000000"
            if len(time_str) == 4:  # HHMM
                time_str += "00"
            dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
            return dt.strftime("%Y:%m:%d %H:%M:%S")
        # Try M-D-YYYY, D-M-YYYY fallback
        fallback = re.match(r"(\d{1,2})[\-_](\d{1,2})[\-_](\d{2,4})", name)
        if fallback:
            m, d, y = fallback.groups()
            if len(y) == 2:
                y = "20" + y  # assume 20xx
            try:
                dt = datetime.strptime(f"{m}-{d}-{y}", "%m-%d-%Y")
                return dt.strftime("%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
            try:
                dt = datetime.strptime(f"{d}-{m}-{y}", "%d-%m-%Y")
                return dt.strftime("%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        raise ValueError("No valid date format found in filename.")
    except Exception as e:
        print(f"Could not extract date from filename {filename}: {e}")
        return None


def calculate_age(dob_str: str, video_date: datetime) -> Optional[float]:
    """Calculate age in months at time of video.

    Computes the participant's age in months at the time the video was recorded
    based on their date of birth and the video recording date.

    Args:
        dob_str (str): Date of birth string in parseable format.
        video_date (datetime): Date when the video was recorded.

    Returns:
        float or None: Age in months (rounded to 1 decimal place),
                       or None if calculation fails.

    Note:
        Uses 30.44 days per month for calculation to account for
        varying month lengths.

    Example:
        >>> from datetime import datetime
        >>> dob = "2022-01-15"
        >>> video_dt = datetime(2023, 1, 15)
        >>> calculate_age(dob, video_dt)
        12.0
    """
    try:
        dob = parser.parse(dob_str)
        delta = video_date - dob
        age_months = round(delta.days / 30.44, 1)
        return age_months
    except Exception:
        return None


def stabilize_video(input_path: str, stabilized_path: str) -> None:
    """Stabilize video using ffmpeg vidstab.

    Applies video stabilization using FFmpeg's vidstab filter to reduce
    camera shake and improve video quality for analysis.

    Args:
        input_path (str): Path to input video file.
        stabilized_path (str): Path for output stabilized video file.

    Note:
        This function uses a two-pass approach: first detecting motion
        vectors, then applying stabilization transforms. Temporary
        transform files are automatically cleaned up.

    Todo:
        Add error handling for FFmpeg execution failures.
    """
    detect_cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-vf",
        "vidstabdetect=shakiness=5:accuracy=15",
        "-f",
        "null",
        "-",
    ]
    subprocess.run(detect_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    transform_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        "vidstabtransform=smoothing=30:input=transforms.trf",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "23",
        "-c:a",
        "copy",
        stabilized_path,
    ]
    subprocess.run(transform_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists("transforms.trf"):
        os.remove("transforms.trf")


def preprocess_video(input_path: str, output_path: str) -> None:
    """Preprocess video with stabilization, denoising, and standardization.

    Applies a comprehensive video processing pipeline including stabilization,
    denoising, color equalization, and format standardization to prepare
    videos for behavioral analysis.

    Args:
        input_path (str): Path to input video file.
        output_path (str): Path for output processed video file.

    Note:
        The processing pipeline includes:
        - Video stabilization using vidstab
        - Deinterlacing using yadif
        - Noise reduction using hqdn3d
        - Color equalization
        - Resolution scaling to 720p
        - Frame rate standardization
        - H.264 encoding with optimized settings

    Todo:
        Add progress reporting for long video processing tasks.
    """
    stabilized_tmp = input_path.replace(".mp4", "_stab.mp4").replace(
        ".mov", "_stab.mov"
    )
    stabilize_video(input_path, stabilized_tmp)
    vf_filters = (
        "yadif,"
        "hqdn3d,"
        "eq=contrast=1.0:brightness=0.0:saturation=1.0,"
        "scale=-2:720,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,"
        f"fps={TARGET_FRAMERATE}"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        stabilized_tmp,
        "-vf",
        vf_filters,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(stabilized_tmp)


def extract_audio(input_path: str, output_audio_path: str) -> None:
    """Extract audio from video file.

    Extracts audio track from processed video and converts it to standardized
    format suitable for speech and audio analysis.

    Args:
        input_path (str): Path to input video file.
        output_audio_path (str): Path for output audio file.

    Note:
        Audio is extracted with the following specifications:
        - Sample rate: 16 kHz
        - Channels: Mono (1 channel)
        - Encoding: 16-bit PCM WAV
        These settings are optimized for speech analysis applications.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_audio_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_appledouble_metadata(metafile_path: str) -> Dict[str, Any]:
    """Parse AppleDouble metadata files.

    Extracts metadata from macOS AppleDouble files (._filename) which contain
    extended attributes, resource forks, and other file system metadata.

    Args:
        metafile_path (str): Path to AppleDouble metadata file.

    Returns:
        dict: Dictionary containing parsed metadata including extended attributes,
              resource fork information, and Finder info when available.

    Note:
        AppleDouble files are created by macOS when files are copied to
        non-HFS+ filesystems. They preserve metadata that would otherwise
        be lost, including creation dates and extended attributes.

    Example:
        >>> metadata = parse_appledouble_metadata('._video.mp4')
        >>> print(metadata.get('extended_attributes', {}))
    """
    try:
        with open(metafile_path, "rb") as f:
            content = f.read()
        if not content.startswith(b"\x00\x05\x16\x07"):
            return {"info": "Not AppleDouble format"}
        entries = {}
        entry_count = struct.unpack(">H", content[24:26])[0]
        for i in range(entry_count):
            entry_offset = 26 + (i * 12)
            entry_id, offset, length = struct.unpack(
                ">III", content[entry_offset : entry_offset + 12]
            )
            entry_data = content[offset : offset + length]
            # Extended attributes
            if entry_id == 9:
                if b"bplist" in entry_data:
                    try:
                        plist_start = entry_data.index(b"bplist")
                        plist_data = entry_data[plist_start:]
                        xattrs = plistlib.loads(plist_data)
                        for key, val in xattrs.items():
                            if isinstance(val, bytes):
                                try:
                                    val = plistlib.loads(val)
                                except Exception:
                                    val = val.decode(errors="ignore")
                            key_str = key.decode() if isinstance(key, bytes) else key
                            entries[key_str] = val
                    except Exception as e:
                        entries["extended_attributes_error"] = str(e)
            elif entry_id == 2:
                entries["resource_fork_bytes"] = len(entry_data)
            elif entry_id == 1:
                entries["finder_info_present"] = True
        if not entries:
            return {
                "info": "AppleDouble metadata detected",
                "hex_preview": content[:64].hex(),
            }
        return entries
    except Exception as e:
        return {"error": f"Failed to parse AppleDouble: {e}"}


def create_events_tsv(video_metadata: Dict[str, Any], output_path: str) -> None:
    """Create events.tsv file for video.

    Generates a BIDS-compliant events file documenting the timing and nature
    of events in the video session.

    Args:
        video_metadata (dict): Video metadata containing duration information.
        output_path (str): Path for output events TSV file.

    Note:
        For free play sessions, creates a single event spanning the entire
        video duration with trial_type 'free_play'.

    Raises:
        IOError: If unable to write the events file.
    """
    events_data = [
        {
            "onset": 0.0,
            "duration": video_metadata.get("duration_sec", 0),
            "trial_type": "free_play",
            "response_time": "n/a",
        }
    ]

    events_df = pd.DataFrame(events_data)
    events_df.to_csv(output_path, sep="\t", index=False)


def create_video_metadata_json(
    metadata: Dict[str, Any], processing_info: Dict[str, Any], output_path: str
) -> None:
    """Create JSON metadata file for processed video.

    Generates a BIDS-compliant JSON sidecar file containing video metadata,
    processing parameters, and task information.

    Args:
        metadata (dict): Original video metadata from ffprobe.
        processing_info (dict): Information about processing steps applied.
        output_path (str): Path for output JSON metadata file.

    Raises:
        IOError: If unable to write the metadata file.

    Note:
        The JSON file includes both technical specifications and processing
        pipeline information required for reproducible analysis.
    """
    video_json = {
        "TaskName": "free_play",
        "TaskDescription": "Free play session recorded at home",
        "Instructions": "Natural play behavior in home environment",
        "SamplingFrequency": TARGET_FRAMERATE,
        "Resolution": TARGET_RESOLUTION,
        "ProcessingPipeline": {
            "Stabilization": processing_info.get("has_stabilization", False),
            "Denoising": processing_info.get("has_denoising", False),
            "Equalization": processing_info.get("has_equalization", False),
            "StandardizedFPS": TARGET_FRAMERATE,
            "StandardizedResolution": TARGET_RESOLUTION,
        },
        "OriginalMetadata": metadata,
    }

    with open(output_path, "w") as f:
        json.dump(video_json, f, indent=4)


def create_audio_metadata_json(duration_sec: float, output_path: str) -> None:
    """Create JSON metadata file for extracted audio.

    Generates a BIDS-compliant JSON sidecar file for audio files extracted
    from video sessions, documenting technical specifications and task context.

    Args:
        duration_sec (float): Duration of audio file in seconds.
        output_path (str): Path for output JSON metadata file.

    Raises:
        IOError: If unable to write the metadata file.

    Note:
        Audio specifications are standardized for speech analysis:
        16kHz sampling rate, mono channel, 16-bit encoding.
    """
    audio_json = {
        "SamplingFrequency": 16000,
        "Channels": 1,
        "SampleEncoding": "16bit",
        "Duration": duration_sec,
        "TaskName": "free_play",
        "TaskDescription": "Audio extracted from free play session",
    }

    with open(output_path, "w") as f:
        json.dump(audio_json, f, indent=4)


def process_videos(
    video_root: str, demographics_df: pd.DataFrame
) -> Tuple[List[Dict[str, Any]], List[Union[str, Dict[str, Any]]]]:
    """Process videos and organize in BIDS format.

    Main processing function that walks through video directories, processes
    each video file, and organizes the results according to BIDS specification.

    Args:
        video_root (str): Root directory containing video files.
        demographics_df (pd.DataFrame): DataFrame containing participant demographics.

    Returns:
        tuple: A tuple containing:
            - list: Successfully processed video entries with metadata
            - list: Videos that failed processing with error information
                   (strings for simple failures, dicts for detailed errors)

    Note:
        This function performs the complete processing pipeline:
        1. Video discovery and metadata extraction
        2. Participant identification and matching
        3. BIDS directory structure creation
        4. Video processing (stabilization, denoising, standardization)
        5. Audio extraction
        6. Metadata file generation

    Todo:
        Add parallel processing support for large video collections.
        Implement progress reporting with estimated completion times.
    """
    all_data = []
    not_processed: List[Union[str, Dict[str, Any]]] = []
    processed_files = set()
    demographics_df["dependent_temporary_id"] = (
        demographics_df["dependent_temporary_id"].astype(str).str.upper()
    )

    for root, dirs, files in os.walk(video_root):
        for file in files:
            if file.startswith("._"):
                real_name = file[2:]
                real_path = os.path.join(root, real_name)
                if os.path.exists(real_path):
                    metadata_path = os.path.join(root, file)
                    metadata_info = parse_appledouble_metadata(metadata_path)
                    print(f"[AppleDouble] Metadata for {real_name}: {metadata_info}")
                continue  # Skip ._ file itself

            # Skip unsupported formats
            if not file.lower().endswith((".mov", ".mp4")):
                print(f"[SKIP] Unsupported file type: {file}")
                continue

            if file.lower().endswith((".mov", ".mp4")) and not file.startswith(
                ".DS_Store"
            ):
                if file in processed_files:
                    continue
                processed_files.add(file)
                video_path = os.path.join(root, file)

                try:
                    print(f"[PROCESS] Processing file: {file}")
                    exif_data = extract_exif(video_path)
                    if "error" in exif_data or "ffprobe_error" in exif_data:
                        raise ValueError("Unreadable or unsupported video format")

                    # Extract participant ID from folder structure
                    folder_parts = Path(video_path).parts
                    matching_folder = next(
                        (
                            part
                            for part in folder_parts
                            if "_" in part
                            and part.upper().endswith(
                                tuple(demographics_df["dependent_temporary_id"].values)
                            )
                        ),
                        None,
                    )
                    if not matching_folder:
                        not_processed.append(video_path)
                        continue

                    participant_id_str = matching_folder.split("_")[-1].upper()
                    demo_row = demographics_df[
                        demographics_df["dependent_temporary_id"] == participant_id_str
                    ]
                    if demo_row.empty:
                        not_processed.append(video_path)
                        continue

                    # Create consistent numeric participant ID for BIDS
                    bids_participant_id = f"sub-{hash(participant_id_str) % 10000:04d}"
                    bids_participant_num = hash(participant_id_str) % 10000

                    # Determine session from path
                    session_id = get_session_from_path(video_path)

                    # Extract video date and calculate age
                    video_date_str = extract_date_from_filename(file)
                    if not video_date_str:
                        raise ValueError("Could not extract date from filename")
                    video_date = datetime.strptime(video_date_str, "%Y:%m:%d %H:%M:%S")
                    age = calculate_age(demo_row.iloc[0]["dependent_dob"], video_date)

                    # Create BIDS directory structure for this participant/session
                    raw_subj_dir = os.path.join(
                        BIDS_ROOT, bids_participant_id, f"ses-{session_id}", "beh"
                    )
                    deriv_subj_dir = os.path.join(
                        DERIVATIVES_DIR, bids_participant_id, f"ses-{session_id}", "beh"
                    )
                    os.makedirs(raw_subj_dir, exist_ok=True)
                    os.makedirs(deriv_subj_dir, exist_ok=True)

                    # Create BIDS filenames
                    raw_video_name = create_bids_filename(
                        bids_participant_num, session_id, "beh", "mp4"
                    )
                    processed_video_name = create_bids_filename(
                        bids_participant_num, session_id, "desc-processed_beh", "mp4"
                    )
                    audio_name = create_bids_filename(
                        bids_participant_num, session_id, "audio", "wav"
                    )
                    events_name = create_bids_filename(
                        bids_participant_num, session_id, "events", "tsv"
                    )
                    processed_events_name = create_bids_filename(
                        bids_participant_num, session_id, "desc-processed_events", "tsv"
                    )

                    # File paths
                    raw_video_path = os.path.join(raw_subj_dir, raw_video_name)
                    processed_video_path = os.path.join(
                        deriv_subj_dir, processed_video_name
                    )
                    audio_path = os.path.join(deriv_subj_dir, audio_name)
                    events_path = os.path.join(raw_subj_dir, events_name)
                    processed_events_path = os.path.join(
                        deriv_subj_dir, processed_events_name
                    )

                    # Copy raw video to BIDS structure
                    if not os.path.exists(raw_video_path):
                        shutil.copy2(video_path, raw_video_path)

                    # Process video
                    if not os.path.exists(processed_video_path):
                        preprocess_video(video_path, processed_video_path)

                    # Extract audio
                    if not os.path.exists(audio_path):
                        extract_audio(processed_video_path, audio_path)

                    # Create events files
                    create_events_tsv(exif_data, events_path)
                    # Copy for derivatives
                    create_events_tsv(exif_data, processed_events_path)

                    # Create metadata JSON files
                    processing_info = {
                        "has_stabilization": True,
                        "has_denoising": True,
                        "has_equalization": True,
                    }

                    video_json_path = processed_video_path.replace(".mp4", ".json")
                    create_video_metadata_json(
                        exif_data, processing_info, video_json_path
                    )

                    audio_json_path = audio_path.replace(".wav", ".json")
                    create_audio_metadata_json(
                        exif_data.get("duration_sec", 0), audio_json_path
                    )

                    # Look for associated AppleDouble metadata
                    apple_metadata = None
                    apple_file = os.path.join(os.path.dirname(video_path), f"._{file}")
                    if os.path.exists(apple_file):
                        apple_metadata = parse_appledouble_metadata(apple_file)

                    entry = {
                        "original_participant_id": participant_id_str,
                        "bids_participant_id": bids_participant_id,
                        "session_id": session_id,
                        "original_video": video_path,
                        "raw_video_bids": raw_video_path,
                        "processed_video_bids": processed_video_path,
                        "audio_file_bids": audio_path,
                        "events_file_bids": events_path,
                        "video_date": video_date.isoformat(),
                        "age_months": age,
                        "duration_sec": exif_data.get("duration_sec", 0),
                        "metadata": exif_data,
                        "apple_metadata": apple_metadata,
                        "processing_info": processing_info,
                    }
                    all_data.append(entry)

                except Exception as e:
                    print(f"[ERROR] Failed to process {video_path}: {str(e)}")
                    not_processed.append({"video": video_path, "error": str(e)})

    return all_data, not_processed


def save_json(data: Union[List[Any], Dict[str, Any]], path: str) -> None:
    """Save data to JSON file.

    Utility function to save Python data structures to JSON files with
    proper formatting and error handling.

    Args:
        data (list or dict): Data structure to save as JSON.
        path (str): Output file path for JSON file.

    Raises:
        IOError: If unable to write to the specified path.
        TypeError: If data contains non-serializable objects.

    Note:
        Uses 4-space indentation for readable JSON output.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main() -> None:
    """Main processing function.

    Orchestrates the complete BIDS video processing pipeline including
    directory structure creation, dataset description generation, video
    processing, and metadata file creation.

    This function serves as the entry point for the processing pipeline
    and handles the overall workflow coordination.

    Raises:
        Exception: Various exceptions may be raised during processing,
                  which are caught and reported appropriately.

    Note:
        Processing progress and statistics are printed to stdout for
        monitoring large batch operations.

    Example:
        >>> main()
        Starting BIDS format video processing...
        [PROCESS] Processing file: video001.mp4
        ...
        Processing complete!
        Successfully processed: 45 videos
        Failed to process: 2 videos
    """
    print("Starting BIDS format video processing...")

    # Create BIDS directory structure
    create_bids_structure()

    # Create dataset description files
    create_dataset_description()
    create_derivatives_dataset_description()

    # Create README file
    create_readme()

    # Read demographics and process videos
    demographics_df = read_demographics(ASD_CSV, NONASD_CSV)
    all_data, not_processed = process_videos(VIDEO_ROOT, demographics_df)

    # Create participants files
    create_participants_files(demographics_df, all_data)

    # Save processing logs
    save_json(all_data, os.path.join(OUTPUT_DIR, "bids_processing_log.json"))
    save_json(not_processed, os.path.join(OUTPUT_DIR, "bids_not_processed.json"))

    print("Processing complete!")
    print(f"Successfully processed: {len(all_data)} videos")
    print(f"Failed to process: {len(not_processed)} videos")
    print(f"BIDS dataset created at: {BIDS_ROOT}")


if __name__ == "__main__":
    main()
