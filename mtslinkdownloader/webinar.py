import logging
import os
import re
from typing import Optional

from .downloader import construct_json_data_url, fetch_json_data
from .processor import process_composite_video
from .utils import create_directory_if_not_exists


def fetch_webinar_data(event_sessions: str, record_id: Optional[str] = None, session_id=None,
                       max_duration=None, hide_silent: bool = False,
                       start_time: float = 0, quality: str = "1080p", status_file: Optional[str] = None):
    json_data_url = construct_json_data_url(event_session_id=event_sessions, recording_id=record_id)
    json_data = fetch_json_data(url=json_data_url, session_id=session_id)

    if not json_data:
        logging.error('Failed to fetch webinar data. Check the session ID or URL.')
        return

    sanitized_name = re.sub(r'[\s\/:*?"<>|]+', '_', json_data['name'])
    directory = create_directory_if_not_exists(sanitized_name)
    output_video_path = os.path.join(directory, f'{sanitized_name}.mp4')

    process_composite_video(directory, json_data, output_video_path, max_duration,
                            hide_silent=hide_silent, start_time=start_time, quality=quality,
                            status_file=status_file)
    
    full_path = os.path.abspath(output_video_path)
    logging.info(f'Final video saved to: {full_path}')

    return 1
