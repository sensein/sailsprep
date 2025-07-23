import os
import json
import shutil
import subprocess
import pandas as pd
from datetime import datetime
from moviepy.editor import VideoFileClip
import uuid
import yaml
from dotenv import load_dotenv
from dateutil import parser
from pathlib import Path
import logging
import binascii

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

VIDEO_ROOT = config['video_root']
ASD_CSV = config['asd_csv']
NONASD_CSV = config['nonasd_csv']
OUTPUT_DIR = config['output_dir']
TARGET_RESOLUTION = config.get('target_resolution', '1280x720')
TARGET_FRAMERATE = config.get('target_fps', 30)

# Output directories
PREPROCESSED_VIDEO_DIR = os.path.join(OUTPUT_DIR, 'preprocessed_videos')
AUDIO_DIR = os.path.join(OUTPUT_DIR, 'extracted_audio')
os.makedirs(PREPROCESSED_VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

def read_demographics(asd_csv, nonasd_csv):
    df_asd = pd.read_csv(asd_csv)
    df_nonasd = pd.read_csv(nonasd_csv)
    df = pd.concat([df_asd, df_nonasd], ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'id' in df.columns:
        df['participant_id'] = df['id'].apply(lambda x: f"sub-{str(x).zfill(3)}")
    elif 'participant_id' in df.columns:
        df['participant_id'] = df['participant_id'].apply(lambda x: f"sub-{str(x).zfill(3)}")
    else:
        # Auto-generate IDs if missing
        df['participant_id'] = [f"sub-{str(i+1).zfill(3)}" for i in range(len(df))]

    # Reorder columns to match BIDS expectations
    priority_cols = ['participant_id', 'age', 'sex', 'diagnosis']
    existing_priority = [col for col in priority_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_priority]
    df = df[existing_priority + other_cols]

    # Save to participants.tsv in output_dir
    output_path = os.path.join(OUTPUT_DIR, 'participants.tsv')
    df.to_csv(output_path, sep='\t', index=False)
    return df 

# metadata

def extract_exif(video_path):
   try:
       cmd = [
           "ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", video_path
       ]
       result = subprocess.run(cmd, capture_output=True, text=True)
       if result.returncode != 0:
           return {"ffprobe_error": result.stderr.strip()}
       metadata = json.loads(result.stdout)
       extracted = {}
       # Format-level metadata
       format_info = metadata.get("format", {})
       extracted['filename'] = format_info.get("filename")
       extracted['format'] = format_info.get("format_long_name")
       extracted['duration_sec'] = float(format_info.get("duration", 0))
       extracted['bit_rate'] = int(format_info.get("bit_rate", 0))
       extracted['size_bytes'] = int(format_info.get("size", 0))
       # All date/time-related tags from format
       extracted['format_dates'] = {}
       if 'tags' in format_info:
           for k, v in format_info['tags'].items():
               if 'date' in k.lower() or 'time' in k.lower():
                   extracted['format_dates'][k] = v
       # Loop through all streams (video, audio, etc.)
       extracted['stream_dates'] = []
       for stream in metadata.get("streams", []):
           stream_entry = {}
           if 'tags' in stream:
               for k, v in stream['tags'].items():
                   if 'date' in k.lower() or 'time' in k.lower():
                       stream_entry[k] = v
           if stream_entry:
               extracted['stream_dates'].append(stream_entry)
       return extracted
   except Exception as e:
       return {"error": str(e)}
import re
def extract_date_from_filename(filename):
   import re
   from datetime import datetime
   try:
       name = os.path.splitext(os.path.basename(filename))[0]
       # Try direct known formats
       known_formats = [
           "%m-%d-%Y", "%m-%d-%y", "%m_%d_%Y", "%m_%d_%y",
           "%Y-%m-%d", "%Y%m%d", "%m%d%Y"
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
       fallback = re.match(r"(\\d{1,2})[\\-_](\\d{1,2})[\\-_](\\d{2,4})", name)
       if fallback:
           m, d, y = fallback.groups()
           if len(y) == 2:
               y = "20" + y  # assume 20xx
           try:
               dt = datetime.strptime(f"{m}-{d}-{y}", "%m-%d-%Y")
               return dt.strftime("%Y:%m:%d %H:%M:%S")
           except:
               pass
           try:
               dt = datetime.strptime(f"{d}-{m}-{y}", "%d-%m-%Y")
               return dt.strftime("%Y:%m:%d %H:%M:%S")
           except:
               pass
       raise ValueError("No valid date format found in filename.")
   except Exception as e:
       print(f"Could not extract date from filename {filename}: {e}")
       return None

def calculate_age(dob_str, video_date):
    try:
        dob = parser.parse(dob_str)
        delta = video_date - dob
        age_months = round(delta.days / 30.44, 1)
        return age_months
    except Exception:
        return None

def stabilize_video(input_path, stabilized_path):
    detect_cmd = ["ffmpeg", "-i", input_path, "-vf", "vidstabdetect=shakiness=5:accuracy=15", "-f", "null", "-"]
    subprocess.run(detect_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    transform_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "vidstabtransform=smoothing=30:input=transforms.trf",
        "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-c:a", "copy",
        stabilized_path
    ]
    subprocess.run(transform_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists("transforms.trf"):
        os.remove("transforms.trf")

def preprocess_video(input_path, output_path):
    stabilized_tmp = input_path.replace(".mp4", "_stab.mp4").replace(".mov", "_stab.mov")
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
        "ffmpeg", "-y", "-i", stabilized_tmp,
        "-vf", vf_filters,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(stabilized_tmp)

def extract_audio(input_path, output_audio_path):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


import struct
import plistlib
def parse_appledouble_metadata(metafile_path):
   try:
       with open(metafile_path, 'rb') as f:
           content = f.read()
       if not content.startswith(b'\x00\x05\x16\x07'):
           return {"info": "Not AppleDouble format"}
       entries = {}
       entry_count = struct.unpack(">H", content[24:26])[0]
       for i in range(entry_count):
           entry_offset = 26 + (i * 12)
           entry_id, offset, length = struct.unpack(">III", content[entry_offset:entry_offset+12])
           entry_data = content[offset:offset+length]
           # Extended attributes
           if entry_id == 9:
               if b'bplist' in entry_data:
                   try:
                       plist_start = entry_data.index(b'bplist')
                       plist_data = entry_data[plist_start:]
                       xattrs = plistlib.loads(plist_data)
                       for key, val in xattrs.items():
                           if isinstance(val, bytes):
                               try:
                                   val = plistlib.loads(val)
                               except:
                                   val = val.decode(errors='ignore')
                           entries[key.decode() if isinstance(key, bytes) else key] = val
                   except Exception as e:
                       entries['extended_attributes_error'] = str(e)
           elif entry_id == 2:
               # Optional: include size of resource fork
               entries['resource_fork_bytes'] = len(entry_data)
           elif entry_id == 1:
               entries['finder_info_present'] = True
       if not entries:
           return {"info": "AppleDouble metadata detected", "hex_preview": content[:64].hex()}
       return entries
   except Exception as e:
       return {"error": f"Failed to parse AppleDouble: {e}"}

def process_videos(video_root, demographics_df):
    all_data = []
    not_processed = []
    processed_files = set()
    demographics_df['dependent_temporary_id'] = demographics_df['dependent_temporary_id'].astype(str).str.upper()

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
            # Skip unsupported formats like .3gp
            if not file.lower().endswith(('.mov', '.mp4')):
                print(f"[SKIP] Unsupported file type: {file}")
                continue
            print(f"[PROCESS] Processing file: {file}")

            if file.lower().endswith(('.mov', '.mp4')) and not file.startswith(".DS_Store"):
                if file in processed_files:
                    continue
                processed_files.add(file)
                video_path = os.path.join(root, file)
                try:
                    exif_data = extract_exif(video_path)
                    if 'error' in exif_data or 'ffprobe_error' in exif_data:
                        raise ValueError("Unreadable or unsupported video format")

                    folder_parts = Path(video_path).parts
                    matching_folder = next((part for part in folder_parts if '_' in part and part.upper().endswith(tuple(demographics_df['dependent_temporary_id'].values))), None)
                    if not matching_folder:
                        not_processed.append(video_path)
                        continue
                    participant_id = matching_folder.split("_")[-1].upper()
                    demo_row = demographics_df[demographics_df['dependent_temporary_id'] == participant_id]
                    if demo_row.empty:
                        not_processed.append(video_path)
                        continue

                    video_date_str = extract_date_from_filename(file)
                    if not video_date_str:
                        raise ValueError("Could not extract date from filename")
                    video_date = datetime.strptime(video_date_str, "%Y:%m:%d %H:%M:%S")
                    age = calculate_age(demo_row.iloc[0]['dependent_dob'], video_date)

                    preprocessed_name = f"{participant_id}_{uuid.uuid4().hex[:8]}.mp4"
                    preprocessed_path = os.path.join(PREPROCESSED_VIDEO_DIR, preprocessed_name)
                    if os.path.exists(preprocessed_path):
                        continue
                    preprocess_video(video_path, preprocessed_path)
                    audio_path = os.path.join(AUDIO_DIR, preprocessed_name.replace('.mp4', '.wav'))
                    extract_audio(preprocessed_path, audio_path)

                    # Look for associated AppleDouble metadata
                    apple_metadata = None
                    apple_file = os.path.join(root, f"._{file}")
                    if os.path.exists(apple_file):
                        apple_metadata = parse_appledouble_metadata(apple_file)
                    entry = {
                        'participant_id': participant_id,
                        'original_video': video_path,
                        'preprocessed_video': preprocessed_path,
                        'audio_file': audio_path,
                        'video_date': video_date.isoformat(),
                        'age_months': age,
                        'duration_sec': exif_data.get('duration_sec', 0),
                        'metadata': exif_data,
                        'apple_metadata': apple_metadata,  
                        'is_preprocessed': True,
                        'is_audio_extracted': True,
                        'has_stabilization': True,
                        'has_denoising': True,
                        'has_equalization': True
                    }
                    all_data.append(entry)
                except Exception as e:
                    not_processed.append({'video': video_path, 'error': str(e)})
    return all_data, not_processed

#compiling metadata 

def save_json_metadata(metadata, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)

def save_metadata_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=str) #used for final output 

def preprocess_data(config, demographics_df):
    global PREPROCESSED_VIDEO_DIR, AUDIO_DIR
    PREPROCESSED_VIDEO_DIR = os.path.join(config['output_dir'], 'preprocessed_videos')
    AUDIO_DIR = os.path.join(config['output_dir'], 'extracted_audio')
    os.makedirs(PREPROCESSED_VIDEO_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    all_data, not_processed = process_videos(config['video_root'], demographics_df)

    # Generate metadata_dict
    metadata_dict = {}
    for file in os.listdir(PREPROCESSED_VIDEO_DIR):
        if file.endswith('.mp4'):
            file_path = os.path.join(PREPROCESSED_VIDEO_DIR, file)
            metadata_dict[file] = extract_exif(file_path)
    for file in os.listdir(AUDIO_DIR):
        if file.endswith('.wav'):
            file_path = os.path.join(AUDIO_DIR, file)
            metadata_dict[file] = extract_exif(file_path)

    organize_per_participant(PREPROCESSED_VIDEO_DIR, AUDIO_DIR, metadata_dict)
    return all_data, not_processed

# organizing per participant 
def organize_per_participant(processed_video_dir, processed_audio_dir, metadata_dict):
    for filename in os.listdir(processed_video_dir):
        if filename.endswith('.mp4'):
            participant_id = filename.split('.')[0]  # e.g., "sub-001"
            video_src = os.path.join(processed_video_dir, filename)
            audio_filename = filename.replace('.mp4', '.wav')
            audio_src = os.path.join(processed_audio_dir, audio_filename)

            # Build BIDS-like paths
            bids_dir = os.path.join(OUTPUT_DIR, participant_id, 'ses-01', 'beh')
            os.makedirs(bids_dir, exist_ok=True)

            # Construct BIDS-compliant filenames
            base_name = f"{participant_id}_ses-01_task-default"
            video_dst = os.path.join(bids_dir, f"{base_name}_behvideo.mp4")
            audio_dst = os.path.join(bids_dir, f"{base_name}_behaudio.wav")
            video_json = os.path.join(bids_dir, f"{base_name}_behvideo.json")
            audio_json = os.path.join(bids_dir, f"{base_name}_behaudio.json")

            # Move files
            shutil.copy(video_src, video_dst)
            if os.path.exists(audio_src):
                shutil.copy(audio_src, audio_dst)

            # Save JSON metadata 
            if filename in metadata_dict:
                save_json_metadata(metadata_dict[filename], video_json)
            if audio_filename in metadata_dict:
                save_json_metadata(metadata_dict[audio_filename], audio_json)

metadata_dict = {}
for file in os.listdir(PREPROCESSED_VIDEO_DIR):
    if file.endswith('.mp4'):
        file_path = os.path.join(PREPROCESSED_VIDEO_DIR, file)
        metadata_dict[file] = extract_exif(file_path)

for file in os.listdir(AUDIO_DIR):
    if file.endswith('.wav'):
        file_path = os.path.join(AUDIO_DIR, file)
        metadata_dict[file] = extract_exif(file_path)

organize_per_participant(PREPROCESSED_VIDEO_DIR, AUDIO_DIR, metadata_dict)

# dataset_description.json 
def write_dataset_description(output_dir):
    description = {
        "Name": "ASD vs NonASD Behavioral Dataset",
        "BIDSVersion": "1.8.0",
        "License": "CC-BY-4.0",
        "Authors": ["Your Name"],
        "Acknowledgements": "Thanks to all contributors and families.",
        "HowToAcknowledge": "Please cite: YourPublicationOrDOI",
        "DatasetType": "raw"
    }
    path = os.path.join(output_dir, "dataset_description.json")
    with open(path, 'w') as f:
        json.dump(description, f, indent=4)
        
def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    global OUTPUT_DIR
    OUTPUT_DIR = config['output_dir']

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Read participant info
    demographics_df = read_demographics(config['asd_csv'], config['nonasd_csv'])

    # Step 2: Preprocess and extract audio/video, collect metadata
    metadata, not_processed = preprocess_data(config, demographics_df)

    # Step 3: Save metadata and error logs
    save_metadata_json(metadata, os.path.join(OUTPUT_DIR, "final_metadata.json"))
    save_metadata_json(not_processed, os.path.join(OUTPUT_DIR, "not_processed.json"))

    # Step 4: BIDS description file
    write_dataset_description(OUTPUT_DIR)

    print("✅ BIDS conversion complete.")

if __name__ == "__main__":
    main()