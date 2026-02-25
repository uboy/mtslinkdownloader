import json
import sys
import os
sys.path.append(os.getcwd())
from mtslinkdownloader.processor import parse_event_logs, parse_presentation_timeline, compute_layout_timeline, _materialize_slide_timeline, parse_chat_and_questions

def main():
    from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data
    from mtslinkdownloader.cli import extract_ids_from_url
    url = "https://hse.mts-link.ru/j/21462290/13201551527/record-new/12430577767"
    event_sessions, record_id = extract_ids_from_url(url)
    json_data_url = construct_json_data_url(event_session_id=event_sessions, recording_id=record_id)
    json_data = fetch_json_data(url=json_data_url, session_id=None)
    
    total_duration = float(json_data.get('duration', 0))
    slides, s_timeline = parse_presentation_timeline(json_data)
    streams, admin_id = parse_event_logs(json_data)
    chat, questions = parse_chat_and_questions(json_data)
    all_events = sorted(chat + questions, key=lambda x: x['time'])
    
    # Materialize without real downloads
    s_map = {u: f"path_to_{u[-10:]}" for u in slides}
    materialized_s_timeline = [(s, e, s_map[u]) for s, e, u in s_timeline if u in s_map]
    
    # Simulation
    for sk, s in streams.items():
        for clip in s['clips']:
            if clip['duration'] == 0:
                clip['duration'] = total_duration - clip['relative_time']
            clip['has_video'] = True
    
    timeline = compute_layout_timeline(streams, total_duration, admin_id, False, {}, 0, materialized_s_timeline, all_events)
    
    for i, seg in enumerate(timeline):
        if 5130 <= seg['start'] <= 5150:
            print(f"Seg {i}: {seg['start']:.2f} - {seg['end']:.2f} slide={seg['slide_image']}")

if __name__ == "__main__":
    main()
