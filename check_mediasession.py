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
        if module == 'mediasession.add':
            d = e.get('data', {})
            sd = d.get('stream', {})
            is_ss = 'screensharing' in sd
            if not is_ss:
                st = str(sd.get('type', '')).lower()
                sn = str(sd.get('name', '')).lower()
                if any(k in st or k in sn for k in ('screen', 'presentation', 'desktop')):
                    is_ss = True
            
            if is_ss:
                print(f"[{e.get('relativeTime'):.2f}] mediasession.add: SCREENSHARE url={d.get('url')}")

if __name__ == "__main__":
    main()
