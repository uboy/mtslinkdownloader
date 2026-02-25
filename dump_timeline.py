import json
import sys
import os
sys.path.append(os.getcwd())
from mtslinkdownloader.processor import parse_presentation_timeline

def main():
    from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data
    from mtslinkdownloader.cli import extract_ids_from_url
    url = "https://hse.mts-link.ru/j/21462290/13201551527/record-new/12430577767"
    event_sessions, record_id = extract_ids_from_url(url)
    json_data_url = construct_json_data_url(event_session_id=event_sessions, recording_id=record_id)
    json_data = fetch_json_data(url=json_data_url, session_id=None)
    
    slides, timeline = parse_presentation_timeline(json_data)
    
    for i, (s, e, u) in enumerate(timeline):
        if 4500 <= s <= 5600 or 4500 <= e <= 5600:
            print(f"[{i}] {s:.2f} - {e:.2f}: {u}")

if __name__ == "__main__":
    main()
