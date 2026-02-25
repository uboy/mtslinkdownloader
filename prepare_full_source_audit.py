import os
import sys
import json
import logging
from typing import Optional, List, Tuple

sys.path.append(os.getcwd())

from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data, download_video_chunk, create_shared_client
from mtslinkdownloader.processor import (
    parse_presentation_timeline, 
    parse_event_logs, 
    parse_chat_and_questions,
    _materialize_slide_timeline,
    compute_layout_timeline
)
from mtslinkdownloader.cli import extract_ids_from_url

def main():
    url = "https://hse.mts-link.ru/j/21462290/13201551527/record-new/12430577767"
    audit_dir = "full_audit_data"
    if not os.path.exists(audit_dir): os.makedirs(audit_dir)

    print("--- FETCHING METADATA ---")
    event_sessions, record_id = extract_ids_from_url(url)
    json_data_url = construct_json_data_url(event_session_id=event_sessions, recording_id=record_id)
    json_data = fetch_json_data(url=json_data_url, session_id=None)
    
    total_duration = float(json_data.get('duration', 0))
    print(f"Total Webinar Duration: {total_duration:.2f}s")

    # 1. Parse everything
    slides, s_timeline = parse_presentation_timeline(json_data)
    streams, admin_id = parse_event_logs(json_data)
    chat, questions = parse_chat_and_questions(json_data)
    all_events = sorted(chat + questions, key=lambda x: x['time'])

    # 2. Download Slides
    print("\n--- DOWNLOADING SLIDES ---")
    materialized_slides = _materialize_slide_timeline(audit_dir, slides, s_timeline)

    # 3. Download Video Chunks
    print("\n--- DOWNLOADING VIDEO CHUNKS (Screenshare & Cameras) ---")
    all_video_urls = set()
    for sk, s in streams.items():
        for clip in s['clips']:
            all_video_urls.add(clip['url'])
    
    print(f"Found {len(all_video_urls)} unique video fragments.")
    
    with create_shared_client() as client:
        urls_list = list(all_video_urls)
        for i, v_url in enumerate(urls_list):
            if i % 20 == 0: print(f"Downloading video {i}/{len(urls_list)}...")
            try:
                download_video_chunk(v_url, audit_dir, client=client)
            except Exception as e:
                print(f"Failed to download {v_url}: {e}")

    # 4. Generate Master Manifest
    print("\n--- GENERATING MASTER MANIFEST ---")
    for sk, s in streams.items():
        for clip in s['clips']:
            if clip['duration'] == 0: clip['duration'] = total_duration - clip['relative_time']
            clip['has_video'] = True

    timeline = compute_layout_timeline(streams, total_duration, admin_id, False, {}, 0, materialized_slides, all_events)

    manifest_path = os.path.join(audit_dir, "MASTER_MANIFEST.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"FULL AUDIT MANIFEST FOR WEBINAR {record_id}\n")
        f.write("="*80 + "\n\n")
        for i, seg in enumerate(timeline):
            f.write(f"SEG {i:04d} | {seg['start']:>10.2f}s -> {seg['end']:>10.2f}s\n")
            if seg.get('screenshare'):
                sk, clip, seek = seg['screenshare']
                f.write(f"  [MAIN] SCREENSHARE: {os.path.basename(clip['url'])} (Seek: {seek:.2f}s)\n")
            elif seg.get('slide_image'):
                f.write(f"  [MAIN] SLIDE: {os.path.basename(seg['slide_image'])}\n")
            else:
                f.write(f"  [MAIN] EMPTY\n")
            
            cams = [streams[sk[0]]['user_name'] for sk in seg.get('cameras', [])]
            f.write(f"  [CAMS] {', '.join(cams) if cams else 'None'}\n")
            
            # Chat check
            t_mid = (seg['start'] + seg['end']) / 2
            seg_chat = [e for e in all_events if e['time'] <= t_mid]
            last_msg = seg_chat[-1]['text'][:50].replace('\n', ' ') if seg_chat else "None"
            f.write(f"  [CHAT] Count: {len(seg_chat)} | Last msg: {last_msg}\n")
            f.write("-" * 40 + "\n")

    print(f"\nAudit data ready in '{audit_dir}'")

if __name__ == "__main__":
    main()
