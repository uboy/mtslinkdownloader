import json
import sys
import os
sys.path.append(os.getcwd())

def main():
    from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data
    from mtslinkdownloader.cli import extract_ids_from_url
    url = "https://hse.mts-link.ru/j/21462290/13201551527/record-new/12430577767"
    event_sessions, record_id = extract_ids_from_url(url)
    json_data_url = construct_json_data_url(event_session_id=event_sessions, recording_id=record_id)
    json_data = fetch_json_data(url=json_data_url, session_id=None)
    events = json_data.get('eventLogs', [])
    
    for e in events:
        module = str(e.get('module', ''))
        if 'screen' in module.lower() or 'media' in module.lower() or 'stream' in module.lower():
            t = e.get('relativeTime', 0)
            if 4000 <= t <= 6000:
                 print(f"[{t:.2f}] {module}: {e.get('data')}")

if __name__ == "__main__":
    main()
