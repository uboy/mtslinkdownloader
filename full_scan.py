import os
import sys
import json
import logging
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
from investigate_sources import extract_frame

def format_time_readable(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_time_filename(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}-{m:02d}-{s:02d}_{int(seconds)}"

def main(url: str, session_id: Optional[str] = None):
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

    debug_dir = "full_webinar_analysis"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    total_duration = float(json_data.get('duration', 0))
    print(f"Total duration: {total_duration:.2f}s ({format_time_readable(total_duration)})")

    # Parse presentation and streams
    slides, s_timeline = parse_presentation_timeline(json_data)
    streams, admin_id = parse_event_logs(json_data)
    
    # Estimate durations
    for sk, s in streams.items():
        for clip in s['clips']:
            if clip['duration'] == 0:
                clip['duration'] = total_duration - clip['relative_time']
            clip['has_video'] = True

    print("\nScanning for all screenshare intervals...")
    screenshare_intervals = []
    for sk, s in streams.items():
        if s['is_screenshare']:
            for clip in s['clips']:
                screenshare_intervals.append({
                    'start': clip['relative_time'],
                    'end': clip['relative_time'] + clip['duration'],
                    'url': clip['url'],
                    'user': s.get('user_name', 'Unknown'),
                    'sk': sk
                })
    
    screenshare_intervals.sort(key=lambda x: x['start'])

    report_path = os.path.join(debug_dir, "full_timeline.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'Start':>10} | {'End':>10} | {'Type':<12} | {'Description'}\n")
        f.write("-" * 100 + "\n")
        
        with create_shared_client() as client:
            for ss in screenshare_intervals:
                f.write(f"{format_time_readable(ss['start']):>10} | {format_time_readable(ss['end']):>10} | Screenshare  | User: {ss['user']}, URL: {ss['url']}\n")
                
                mid_time = (ss['start'] + ss['end']) / 2
                seek = mid_time - ss['start']
                
                print(f"Processing screenshare at {format_time_readable(ss['start'])}...")
                try:
                    video_path = download_video_chunk(ss['url'], debug_dir, client=client)
                    frame_name = f"ss_{format_time_filename(ss['start'])}.jpg"
                    frame_path = os.path.join(debug_dir, frame_name)
                    if not os.path.exists(frame_path):
                        extract_frame(video_path, seek, frame_path)
                except Exception as e:
                    print(f"  Failed: {e}")

        f.write("\n" + "="*20 + " PRESENTATION SLIDES " + "="*20 + "\n")
        for start, end, url in s_timeline:
            f.write(f"{format_time_readable(start):>10} | {format_time_readable(end):>10} | Slide        | {url}\n")

    print(f"\nFull analysis complete. Results in '{debug_dir}'.")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    webinar_url = "https://hse.mts-link.ru/j/21462290/13201551527/record-new/12430577767"
    main(webinar_url)
