import os
import sys
import json
import logging
import re
from typing import Optional, Tuple, List, Dict

# Add current directory to sys.path to import from mtslinkdownloader
sys.path.append(os.getcwd())

from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data, create_shared_client
from mtslinkdownloader.processor import (
    parse_presentation_timeline, 
    parse_event_logs, 
    _find_clip_at_time,
    _choose_primary_screenshare
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

def download_slides_with_timestamps(directory: str, slide_timeline: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """Downloads slides and gives them names with timestamps."""
    materialized = []
    url_to_path = {}
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    with create_shared_client() as client:
        for start, end, url in slide_timeline:
            if url not in url_to_path:
                ts_str = format_time_filename(start)
                filename = f"slide_{ts_str}.jpg"
                path = os.path.join(directory, filename)
                
                if not os.path.exists(path):
                    try:
                        response = client.get(url)
                        response.raise_for_status()
                        with open(path, 'wb') as f:
                            f.write(response.content)
                        # print(f"Downloaded: {filename}")
                    except Exception as e:
                        print(f"Failed to download {url}: {e}")
                        path = f"MISSING_{url}"
                
                url_to_path[url] = path
            
            materialized.append((start, end, url_to_path[url]))
            
    return materialized

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

    debug_dir = "debug_slides_output"
    total_duration = float(json_data.get('duration', 0))
    print(f"Total duration: {total_duration:.2f}s ({format_time_readable(total_duration)})")

    # Parse presentation and streams
    slides, s_timeline = parse_presentation_timeline(json_data)
    streams, admin_id = parse_event_logs(json_data)
    
    # We need to estimate durations of clips because we don't want to download them all
    for sk, s in streams.items():
        for clip in s['clips']:
            if clip['duration'] == 0:
                clip['duration'] = total_duration - clip['relative_time']
            clip['has_video'] = True
            clip['has_audio'] = True

    # Materialize slides
    print("\nDownloading and naming slides...")
    materialized_timeline = download_slides_with_timestamps(debug_dir, s_timeline)

    # Calculate combined timeline (Slides + Screenshares)
    combined_timeline = []
    
    # Add slides
    for start, end, path in materialized_timeline:
        combined_timeline.append({'start': start, 'end': end, 'type': 'Slide', 'label': os.path.basename(path)})
    
    # Add screenshares
    for sk, s in streams.items():
        if s['is_screenshare']:
            for clip in s['clips']:
                combined_timeline.append({
                    'start': clip['relative_time'], 
                    'end': clip['relative_time'] + clip['duration'], 
                    'type': 'Screenshare', 
                    'label': f"StreamID {sk[0]} ({s.get('user_name', 'Unknown')})"
                })
    
    combined_timeline.sort(key=lambda x: x['start'])

    # Save the enhanced report
    report_path = os.path.join(debug_dir, "timeline_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'Start (HMS)':>10} | {'End (HMS)':>10} | {'Type':<12} | {'Label'}\n")
        f.write("-" * 80 + "\n")
        for entry in combined_timeline:
            f.write(f"{format_time_readable(entry['start']):>10} | {format_time_readable(entry['end']):>10} | {entry['type']:<12} | {entry['label']}\n")
    
    print(f"Enhanced timeline report saved to: {report_path}")

if __name__ == "__main__":
    webinar_url = "https://hse.mts-link.ru/j/21462290/13201551527/record-new/12430577767"
    main(webinar_url)
