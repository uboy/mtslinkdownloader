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
    
    overlaps = []
    for i in range(len(timeline)-1):
        s1, e1, u1 = timeline[i]
        s2, e2, u2 = timeline[i+1]
        if s2 < e1:
            overlaps.append((i, i+1, s1, e1, s2, e2))
    
    print(f"Total overlaps found: {len(overlaps)}")
    for o in overlaps[:10]:
        print(f"Overlap between {o[0]} and {o[1]}: [{o[2]:.2f}, {o[3]:.2f}] and [{o[4]:.2f}, {o[5]:.2f}]")

if __name__ == "__main__":
    main()
