import json
import sys
import os
sys.path.append(os.getcwd())
from mtslinkdownloader.processor import _extract_slide_url

def main():
    from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data
    from mtslinkdownloader.cli import extract_ids_from_url
    url = "https://hse.mts-link.ru/j/21462290/13201551527/record-new/12430577767"
    event_sessions, record_id = extract_ids_from_url(url)
    json_data_url = construct_json_data_url(event_session_id=event_sessions, recording_id=record_id)
    json_data = fetch_json_data(url=json_data_url, session_id=None)
    events = json_data.get('eventLogs', [])

    slides = []
    current_url = None
    
    for e in events:
        t = e.get('relativeTime', 0)
        module = str(e.get('module', ''))
        if module.startswith('presentation.'):
            d = e.get('data', {})
            if isinstance(d, dict):
                new_slides = d.get('fileReference', {}).get('file', {}).get('slides', [])
                if isinstance(new_slides, list):
                    urls = [s.get('url', '') for s in new_slides if isinstance(s, dict) and s.get('url')]
                    slides = list(dict.fromkeys(slides + urls))
                
                url_val = _extract_slide_url(d, slides, current_url)
                if url_val != current_url:
                    if 4500 <= t <= 5200:
                        print(f"[{t:.2f}] {module}: NEW URL={url_val}")
                    current_url = url_val

if __name__ == "__main__":
    main()
