import json
import sys
import os
sys.path.append(os.getcwd())
from mtslinkdownloader.processor import parse_presentation_timeline, _collect_slide_urls

def main():
    from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data
    from mtslinkdownloader.cli import extract_ids_from_url
    url = "https://hse.mts-link.ru/j/21462290/13201551527/record-new/12430577767"
    event_sessions, record_id = extract_ids_from_url(url)
    json_data_url = construct_json_data_url(event_session_id=event_sessions, recording_id=record_id)
    json_data = fetch_json_data(url=json_data_url, session_id=None)
    
    slides, timeline = parse_presentation_timeline(json_data)
    slide_urls = _collect_slide_urls(slides, timeline)
    
    print(f"Total unique slide URLs: {len(slide_urls)}")
    
    # Check if they exist in debug_slides_output (if I ran it earlier)
    # Or just check if they are valid URLs.
    
    # Wait! I want to see if any Slide in the timeline was missing from slide_map.
    # I'll simulate _download_slide_images
    
    import httpx
    with httpx.Client() as client:
        for i, url in enumerate(slide_urls):
            try:
                # Just a HEAD request to check if it's there
                resp = client.head(url)
                if resp.status_code != 200:
                    print(f"FAILED: Slide {i} URL={url} status={resp.status_code}")
            except Exception as e:
                print(f"ERROR: Slide {i} URL={url} err={e}")

if __name__ == "__main__":
    main()
