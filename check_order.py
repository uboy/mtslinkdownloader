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
    
    pres_events = [e for e in events if str(e.get('module', '')).startswith('presentation.')]
    
    times = [e.get('relativeTime', 0) for e in pres_events]
    is_sorted = all(times[i] <= times[i+1] for i in range(len(times)-1))
    
    print(f"Presentation events sorted: {is_sorted}")
    if not is_sorted:
        for i in range(len(times)-1):
            if times[i] > times[i+1]:
                print(f"Out of order at index {i}: {times[i]} > {times[i+1]} ({pres_events[i].get('module')} vs {pres_events[i+1].get('module')})")

if __name__ == "__main__":
    main()
