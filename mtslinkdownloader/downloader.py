import os
from typing import Dict, Optional, Union

import httpx
import tqdm
import logging

TIMEOUT_SETTINGS = httpx.Timeout(None, connect=None)
CONNECTION_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=8)


def construct_json_data_url(event_session_id: str, recording_id: str) -> str:
    if not event_session_id:
        raise ValueError('Missing webinar event session ID.')
    
    if not recording_id:
        return f'https://my.mts-link.ru/api/eventsessions/{event_session_id}/record?withoutCuts=false'
    return f'https://my.mts-link.ru/api/event-sessions/{event_session_id}/record-files/{recording_id}/flow?withoutCuts=false'


def fetch_json_data(url: str, session_id: Union[str, None]) -> Dict:
    cookies = {}
    if session_id:
        cookies['sessionId'] = session_id

    with httpx.Client(timeout=TIMEOUT_SETTINGS) as client:
        response = client.get(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            },
            cookies=cookies
        )
        
    try:
        error_data = response.json()
        if error_data.get("error", {}).get("code") == 403:
            logging.error(
                'Access denied: session_id token is required. '
                'Provide it using the "--session-id" parameter.'
            )
            return
    except Exception:
        logging.warning('Server response does not contain JSON.')
            
    response.raise_for_status()
    return response.json()


def create_shared_client() -> httpx.Client:
    """Create a shared httpx client with connection pooling for reuse across downloads."""
    return httpx.Client(timeout=TIMEOUT_SETTINGS, limits=CONNECTION_LIMITS)


def download_video_chunk(video_url: str, save_directory: str, client: Optional[httpx.Client] = None) -> str:
    filename = os.path.basename(video_url)
    file_path = os.path.join(save_directory, filename)

    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=TIMEOUT_SETTINGS, limits=CONNECTION_LIMITS)

    try:
        # Check if file exists and verify its size
        if os.path.exists(file_path):
            try:
                # Use HEAD request to check remote file size without downloading
                response = client.head(video_url)
                response.raise_for_status()
                remote_size = int(response.headers.get('content-length', 0))
                local_size = os.path.getsize(file_path)
                
                if remote_size > 0 and local_size == remote_size:
                    return file_path
                
                logging.warning(
                    f'Size mismatch for {filename}: local {local_size} vs remote {remote_size}. '
                    f'Redownloading...'
                )
            except Exception as e:
                logging.warning(f'Could not verify size for {filename}, skipping check: {e}')
                return file_path

        # If we get here, we need to download (or redownload)
        with client.stream('GET', video_url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(file_path, 'wb') as file:
                with tqdm.tqdm(total=total_size, unit='B', unit_scale=True,
                               desc=f'Downloading {filename}') as progress:
                    for chunk in response.iter_bytes(chunk_size=262144):
                        file.write(chunk)
                        progress.update(len(chunk))
    except Exception:
        # Clean up partial file on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    finally:
        if owns_client:
            client.close()
    return file_path
