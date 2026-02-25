import json
import sys
import os
sys.path.append(os.getcwd())
from mtslinkdownloader.processor import parse_event_logs, _reclassify_screenshare_by_dimensions, download_and_probe_all
from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data
from mtslinkdownloader.cli import extract_ids_from_url

def main():
    url = "https://hse.mts-link.ru/j/21462290/13201551527/record-new/12430577767"
    event_sessions, record_id = extract_ids_from_url(url)
    json_data_url = construct_json_data_url(event_session_id=event_sessions, recording_id=record_id)
    json_data = fetch_json_data(url=json_data_url, session_id=None)
    
    streams, admin_id = parse_event_logs(json_data)
    
    # We need to probe dimensions to trigger reclassification
    # But downloading takes time. Let's just mock it for a few clips if we can find them.
    # Actually, I'll just check what parse_event_logs says about them initially.
    
    for sk, s in streams.items():
        print(f"Stream {sk}: user={s['user_name']} is_ss={s['is_screenshare']}")

if __name__ == "__main__":
    main()
