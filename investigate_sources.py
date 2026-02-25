import os
import sys
import json
import logging
import subprocess
from typing import Optional, Tuple, List, Dict

# Add current directory to sys.path to import from mtslinkdownloader
sys.path.append(os.getcwd())

from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data, create_shared_client, download_video_chunk
from mtslinkdownloader.processor import (
    parse_presentation_timeline, 
    parse_event_logs, 
    _find_clip_at_time,
    _choose_primary_screenshare,
    _get_ffmpeg
)
from mtslinkdownloader.cli import extract_ids_from_url

def format_time_filename(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}-{m:02d}-{s:02d}_{int(seconds)}"

def format_time_readable(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def extract_frame(video_path: str, timestamp_s: float, output_path: str):
    ffmpeg = _get_ffmpeg()
    cmd = [
        ffmpeg, '-y',
        '-ss', str(timestamp_s),
        '-i', video_path,
        '-frames:v', '1',
        '-q:v', '2',
        output_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"FFmpeg failed: {e}")
        return False

def main(url: str, target_time: float = 2376.5, session_id: Optional[str] = None):
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
    event_sessions, record_id = extract_ids_from_url(url)
    if not event_sessions:
        print("Failed to parse URL")
        return

    json_data_url = construct_json_data_url(event_session_id=event_sessions, recording_id=record_id)
    json_data = fetch_json_data(url=json_data_url, session_id=session_id)
    if not json_data:
        print("Failed to fetch JSON data")
        return

    debug_dir = "debug_sources_output"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    total_duration = float(json_data.get('duration', 0))
    print(f"Total duration: {total_duration:.2f}s ({format_time_readable(total_duration)})")

    # Parse presentation and streams
    slides, s_timeline = parse_presentation_timeline(json_data)
    streams, admin_id = parse_event_logs(json_data)
    
    # We need a proper duration for simulation if not probed
    for sk, s in streams.items():
        for clip in s['clips']:
            if clip['duration'] == 0:
                clip['duration'] = total_duration - clip['relative_time']
            clip['has_video'] = True # Assume video exists for analysis

    print(f"\n--- Investigating Sources at {target_time}s ({format_time_readable(target_time)}) ---")
    
    # Check what's happening at target_time
    active_clips = []
    for sk, s in streams.items():
        clip, seek = _find_clip_at_time(s, target_time)
        if clip:
            active_clips.append((sk, clip, seek))

    # Identify Primary Content
    screenshare = _choose_primary_screenshare(active_clips, streams, admin_id)
    slide_at_time = next((url for s, e, url in (s_timeline or []) if s <= target_time < e), None)

    if screenshare:
        sk, clip, seek = screenshare
        print(f"PRIMARY SOURCE: Screenshare by {streams[sk].get('user_name', 'Unknown')}")
        print(f"  Stream ID: {sk[0]}")
        print(f"  Clip URL: {clip['url']}")
        print(f"  Seek time: {seek:.2f}s")
        
        # Download and extract frame for verification
        print("Downloading screenshare clip to extract frame...")
        with create_shared_client() as client:
            try:
                video_path = download_video_chunk(clip['url'], debug_dir, client=client)
                frame_name = f"screenshare_{format_time_filename(target_time)}.jpg"
                frame_path = os.path.join(debug_dir, frame_name)
                if extract_frame(video_path, seek, frame_path):
                    print(f"Successfully extracted frame to: {frame_path}")
                else:
                    print("Failed to extract frame.")
            except Exception as e:
                print(f"Download failed: {e}")
    
    if slide_at_time:
        print(f"PRESENTATION: Slide active at this time: {slide_at_time}")
        if screenshare:
            print("  Note: Presentation is OVERRIDDEN by screenshare.")
    else:
        print("PRESENTATION: No active slide at this time.")

    # List other active cameras
    cameras = [c for c in active_clips if not streams[c[0]]['is_screenshare']]
    if cameras:
        print(f"\nOther active cameras ({len(cameras)}):")
        for sk, clip, seek in cameras:
            user = streams[sk].get('user_name', 'Unknown')
            print(f"  - {user} ({sk[0]})")

    # Generate a mini-report for this window
    window_report = os.path.join(debug_dir, f"report_{int(target_time)}.txt")
    with open(window_report, "w", encoding="utf-8") as f:
        f.write(f"Source Investigation at {target_time} ({format_time_readable(target_time)})\n")
        f.write("="*50 + "\n")
        if screenshare:
            f.write(f"Active Screenshare: {screenshare[1]['url']}\n")
        if slide_at_time:
            f.write(f"Active Slide: {slide_at_time}\n")
        f.write("\nAll Active Streams:\n")
        for sk, clip, seek in active_clips:
            kind = "SS" if sk[1] else "Cam"
            user = streams[sk].get('user_name', 'Unknown')
            f.write(f"  [{kind}] {user:<20} | {sk[0]:<10} | {clip['url']}\n")

if __name__ == "__main__":
    webinar_url = "https://hse.mts-link.ru/j/21462290/13201551527/record-new/12430577767"
    # Testing 39:36 (2376) and 39:43 (2383)
    for t in [2376.5, 2383.0]:
        main(webinar_url, t)
