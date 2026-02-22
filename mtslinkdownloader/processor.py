import json
import logging
import math
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import shutil

import imageio_ffmpeg

from .downloader import download_video_chunk, create_shared_client

# Output dimensions
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 30

# Sidebar layout constants (Mode A: screenshare + cameras)
MAIN_WIDTH = 1600
MAIN_HEIGHT = 1080
SIDEBAR_WIDTH = 320
SIDEBAR_HEIGHT = 270
MAX_SIDEBAR_CAMS = OUTPUT_HEIGHT // SIDEBAR_HEIGHT  # = 4

# Grid configs: n_cameras -> (cols, rows)
GRID_CONFIGS = {
    1: (1, 1),
    2: (2, 1),
    3: (2, 2),
    4: (2, 2),
    5: (3, 2),
    6: (3, 2),
    7: (3, 3),
    8: (3, 3),
    9: (3, 3),
}

# Minimum segment duration (seconds) to avoid tiny segments
MIN_SEGMENT_DURATION = 0.5


# ─── FFmpeg / FFprobe utilities ──────────────────────────────────────────

def _get_ffmpeg():
    """Find ffmpeg: prefer system PATH, fall back to imageio_ffmpeg bundle."""
    path = shutil.which('ffmpeg')
    if path:
        return path
    return imageio_ffmpeg.get_ffmpeg_exe()


def _get_ffprobe():
    """Find ffprobe: prefer system PATH, fall back to same dir as ffmpeg."""
    path = shutil.which('ffprobe')
    if path:
        return path
    ffmpeg = _get_ffmpeg()
    ffprobe = os.path.join(os.path.dirname(ffmpeg), 'ffprobe')
    if os.path.exists(ffprobe + '.exe'):
        return ffprobe + '.exe'
    if os.path.exists(ffprobe):
        return ffprobe
    raise FileNotFoundError('ffprobe not found. Install ffmpeg with ffprobe on your system PATH.')


def _check_nvenc_support() -> bool:
    """Check if ffmpeg supports h264_nvenc hardware acceleration."""
    ffmpeg = _get_ffmpeg()
    try:
        result = subprocess.run([ffmpeg, '-encoders'], capture_output=True, text=True)
        return 'h264_nvenc' in result.stdout
    except Exception:
        return False


def _probe_media(file_path: str) -> dict:
    """Probe a media file for video/audio stream info and duration."""
    ffprobe = _get_ffprobe()
    cmd = [
        ffprobe, '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-show_format', file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        logging.error(f'ffprobe returned invalid output for {file_path}')
        return {
            'has_video': False, 'has_audio': False,
            'width': 0, 'height': 0, 'duration': 0,
        }

    duration = float(data.get('format', {}).get('duration', 0))
    info = {
        'has_video': False,
        'has_audio': False,
        'width': 0,
        'height': 0,
        'duration': max(duration, 0),
    }

    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            info['has_video'] = True
            info['width'] = int(stream.get('width', 0))
            info['height'] = int(stream.get('height', 0))
            # Fallback duration from video stream if format duration missing
            if info['duration'] <= 0 and 'duration' in stream:
                info['duration'] = float(stream['duration'])
        elif stream.get('codec_type') == 'audio':
            info['has_audio'] = True
            if info['duration'] <= 0 and 'duration' in stream:
                info['duration'] = float(stream['duration'])

    return info


def _run_ffmpeg(args: list, desc: str = "", timeout: int = None):
    """Run an ffmpeg command, raising on failure."""
    ffmpeg = _get_ffmpeg()
    cmd = [ffmpeg] + args
    cmd_str = ' '.join(cmd)
    if len(cmd_str) > 300:
        cmd_str = cmd_str[:300] + '...'
    logging.debug(f'ffmpeg {desc}: {cmd_str}')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.error(f'ffmpeg timed out ({desc}) after {timeout}s')
        raise RuntimeError(f'ffmpeg timed out ({desc})')
    if result.returncode != 0:
        logging.error(f'ffmpeg stderr ({desc}): {result.stderr[-1000:]}')
        raise RuntimeError(f'ffmpeg failed ({desc}): {result.stderr[-500:]}')


def _detect_black_video(file_path: str, duration: float) -> bool:
    """Return True if >= 90% of the video is black frames."""
    if duration <= 0:
        return False
    ffmpeg = _get_ffmpeg()
    # Only check the first 30 seconds to speed up the process
    check_duration = min(duration, 30.0)
    cmd = [
        ffmpeg, '-threads', '2', '-t', str(check_duration), '-i', file_path,
        '-vf', 'blackdetect=d=0.1:pix_th=0.10',
        '-an', '-f', 'null', '-'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False
    # Parse black_duration from stderr lines like:
    # [blackdetect @ ...] black_start:0 black_end:5.5 black_duration:5.5
    total_black = 0.0
    for line in result.stderr.splitlines():
        m = re.search(r'black_duration:\s*([\d.]+)', line)
        if m:
            total_black += float(m.group(1))
    return total_black >= duration * 0.9


def _detect_speech_intervals(file_path: str, duration: float) -> list:
    """Detect speech intervals using ffmpeg silencedetect.

    Returns list of (start, end) tuples representing speech intervals.
    """
    if duration <= 0:
        return []
    ffmpeg = _get_ffmpeg()
    cmd = [
        ffmpeg, '-threads', '2', '-i', file_path,
        '-af', 'silencedetect=noise=-35dB:d=0.5',
        '-vn', '-f', 'null', '-'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return [(0, duration)]

    # Parse silence_start / silence_end from stderr
    silence_intervals = []
    silence_start = None
    for line in result.stderr.splitlines():
        m_start = re.search(r'silence_start:\s*([\d.]+)', line)
        m_end = re.search(r'silence_end:\s*([\d.]+)', line)
        if m_start:
            silence_start = float(m_start.group(1))
        if m_end:
            s_end = float(m_end.group(1))
            if silence_start is not None:
                silence_intervals.append((silence_start, s_end))
                silence_start = None

    # If still in silence at end of file
    if silence_start is not None:
        silence_intervals.append((silence_start, duration))

    # Invert silence to get speech intervals
    speech = []
    prev_end = 0.0
    for s_start, s_end in sorted(silence_intervals):
        if s_start > prev_end:
            speech.append((prev_end, s_start))
        prev_end = max(prev_end, s_end)
    if prev_end < duration:
        speech.append((prev_end, duration))

    return speech


def _build_stream_speech_timeline(stream: dict) -> list:
    """Convert per-clip speech intervals to absolute timeline intervals."""
    intervals = []
    for clip in stream['clips']:
        for start, end in clip.get('speech_intervals', []):
            abs_start = clip['relative_time'] + start
            abs_end = clip['relative_time'] + end
            intervals.append((abs_start, abs_end))
    # Merge overlapping intervals
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _has_speech_at(speech_timeline: list, t: float) -> bool:
    """Check if there is speech activity at time t."""
    for start, end in speech_timeline:
        if start <= t <= end:
            return True
        if start > t:
            break
    return False


def _reclassify_screenshare_by_dimensions(streams: dict):
    """Reclassify streams as screenshare if their clips are mostly high-resolution.

    Camera feeds are typically small (e.g. 640x480), while screenshare/presentation
    feeds are typically >= 1280x720.
    """
    to_reclassify = []
    for stream_key, stream in streams.items():
        if stream['is_screenshare']:
            continue
        clips_with_video = [c for c in stream['clips'] if c['has_video'] and c['width'] > 0]
        if not clips_with_video:
            continue
        hd_count = sum(1 for c in clips_with_video if c['width'] >= 1280 and c['height'] >= 720)
        if hd_count > len(clips_with_video) / 2:
            new_key = (stream['conference_id'], True)
            to_reclassify.append((stream_key, new_key))

    for old_key, new_key in to_reclassify:
        if new_key not in streams:
            stream = streams[old_key]
            logging.info(
                f'Reclassifying stream conf={stream["conference_id"]} '
                f'user="{stream["user_name"]}" as screenshare (high-resolution clips)'
            )
            stream['is_screenshare'] = True
            streams[new_key] = stream
            del streams[old_key]


# ─── Step 1: Parse event logs into streams ───────────────────────────────

def parse_event_logs(json_data: dict) -> Tuple[dict, Optional[int]]:
    """Parse eventLogs and group media clips by stream.

    Returns:
        streams: dict keyed by stream_key -> stream info dict
            stream_key = (conference_id, is_screenshare)
        admin_user_id: user ID of the admin (or None)
    """
    event_logs = json_data.get('eventLogs', [])
    admin_user_id = None

    # Get admin from eventsession.start snapshot
    for event in event_logs:
        if event.get('module') == 'eventsession.start':
            snapshot = event.get('snapshot', {}).get('data', {})
            for user_entry in snapshot.get('userlist', []):
                if user_entry.get('role') == 'ADMIN':
                    admin_user_id = user_entry.get('user', {}).get('id')
                    break
            break

    # Build conference metadata from conference.add events
    conf_meta = {}
    for event in event_logs:
        if event.get('module') == 'conference.add':
            d = event.get('data', {})
            conf_id = d.get('id')
            if conf_id:
                user = d.get('user', {})
                conf_meta[conf_id] = {
                    'has_video': d.get('hasVideo', False),
                    'has_audio': d.get('hasAudio', False),
                    'user_id': user.get('id'),
                    'user_name': user.get('nickname', ''),
                    'participation_id': d.get('participationId'),
                }

    # Extract mediasession.add events and group by stream
    streams = {}
    for event in event_logs:
        if event.get('module') != 'mediasession.add':
            continue

        data = event.get('data', {})
        if not isinstance(data, dict) or 'url' not in data:
            continue

        stream_data = data.get('stream', {})
        relative_time = event.get('relativeTime', 0)
        url = data['url']

        # Determine if screensharing - check multiple indicators
        is_screenshare = 'screensharing' in stream_data
        if not is_screenshare:
            stream_type = str(stream_data.get('type', '')).lower()
            stream_name = str(stream_data.get('name', '')).lower()
            for keyword in ('screen', 'presentation', 'desktop'):
                if keyword in stream_type or keyword in stream_name:
                    is_screenshare = True
                    break

        if is_screenshare:
            screen_info = stream_data.get('screensharing', {})
            conf_id = screen_info.get('id')
            if not conf_id:
                conf = stream_data.get('conference', {})
                conf_id = conf.get('id')
            stream_key = (conf_id, True)
        else:
            conf = stream_data.get('conference', {})
            conf_id = conf.get('id')
            stream_key = (conf_id, False)

        if stream_key not in streams:
            meta = conf_meta.get(conf_id, {})
            user_id = meta.get('user_id')
            streams[stream_key] = {
                'conference_id': conf_id,
                'is_screenshare': is_screenshare,
                'user_id': user_id,
                'user_name': meta.get('user_name', ''),
                'conf_has_video': meta.get('has_video', False),
                'conf_has_audio': meta.get('has_audio', False),
                'is_admin': user_id == admin_user_id if user_id and admin_user_id else False,
                'clips': [],
            }

        streams[stream_key]['clips'].append({
            'url': url,
            'relative_time': relative_time,
            'file_path': None,
            'duration': 0,
            'width': 0,
            'height': 0,
            'has_video': False,
            'has_audio': False,
        })

    # Sort clips within each stream by time
    for stream in streams.values():
        stream['clips'].sort(key=lambda c: c['relative_time'])

    total_clips = sum(len(s['clips']) for s in streams.values())
    screenshare_count = sum(1 for s in streams.values() if s['is_screenshare'])
    camera_count = len(streams) - screenshare_count
    logging.info(
        f'Parsed {len(streams)} streams ({camera_count} cameras, '
        f'{screenshare_count} screenshare), {total_clips} clips, '
        f'admin_user_id={admin_user_id}'
    )

    return streams, admin_user_id


def parse_presentation_timeline(json_data: dict) -> Tuple[List[str], List[Tuple[float, float, str]]]:
    """Parse presentation.update events to extract slide URLs and timeline.

    Returns:
        slides: list of slide image URLs (from first active event's file reference)
        slide_timeline: list of (start_time, end_time, slide_url) tuples
    """
    event_logs = json_data.get('eventLogs', [])
    slides = []
    slide_timeline = []

    # Collect presentation.update events in chronological order
    pres_events = []
    for event in event_logs:
        if event.get('module') == 'presentation.update':
            pres_events.append(event)

    if not pres_events:
        return slides, slide_timeline

    # Extract slide URLs from the first event that has file reference
    for event in pres_events:
        data = event.get('data', {})
        file_ref = data.get('fileReference', {})
        file_info = file_ref.get('file', {})
        slide_list = file_info.get('slides', [])
        if slide_list:
            slides = [s.get('url', '') for s in slide_list if s.get('url')]
            break

    if not slides:
        logging.info('Presentation events found but no slide URLs extracted.')
        return slides, slide_timeline

    logging.info(f'Found {len(slides)} presentation slides')

    # Build timeline from events
    slide_index = 0
    current_start = None
    current_url = None

    for event in pres_events:
        data = event.get('data', {})
        t = event.get('relativeTime', 0)
        is_active = data.get('isActive', False)

        if is_active:
            # Close previous interval if open
            if current_start is not None:
                slide_timeline.append((current_start, t, current_url))

            current_url = slides[slide_index % len(slides)]
            current_start = t
            slide_index += 1
        else:
            # Presentation hidden — close current interval
            if current_start is not None:
                slide_timeline.append((current_start, t, current_url))
                current_start = None
                current_url = None

    # Close final open interval with total duration
    if current_start is not None:
        total_duration = float(json_data.get('duration', 0))
        slide_timeline.append((current_start, total_duration, current_url))

    logging.info(
        f'Presentation timeline: {len(slide_timeline)} intervals, '
        f'{slide_index} slide changes'
    )

    return slides, slide_timeline


# ─── Step 2: Download and probe clips ────────────────────────────────────

def _download_and_probe_clip(clip: dict, directory: str, client) -> None:
    """Download a single clip and probe its media info."""
    try:
        file_path = download_video_chunk(clip['url'], directory, client=client)
        clip['file_path'] = file_path
    except Exception as e:
        logging.error(f'Critical failure downloading {clip["url"]}: {e}')
        return

    if not os.path.exists(file_path):
        logging.error(f'File {file_path} missing after download attempt.')
        return

    info = _probe_media(file_path)
    clip['duration'] = info['duration']
    clip['width'] = info['width']
    clip['height'] = info['height']
    clip['has_video'] = info['has_video']
    clip['has_audio'] = info['has_audio']

    if clip['duration'] <= 0:
        logging.warning(
            f'Media clip has zero or invalid duration: {file_path}. '
            f'This may cause timing gaps in the final video.'
        )
    elif clip['duration'] < 0.5:
        logging.warning(
            f'Extremely short clip detected ({clip["duration"]:.3f}s): {file_path}. '
            f'Check if this chunk is complete.'
        )

    # Detect black-screen cameras and mark as no video
    # Skip screenshare clips — presentations can have dark slides
    if clip['has_video'] and clip['duration'] > 0 and not clip.get('_is_screenshare'):
        if _detect_black_video(file_path, clip['duration']):
            logging.info(f'Black video detected, marking has_video=False: {file_path}')
            clip['has_video'] = False

    # Run voice activity detection if requested
    if clip.get('_run_vad') and clip['has_audio'] and clip['duration'] > 0:
        clip['speech_intervals'] = _detect_speech_intervals(file_path, clip['duration'])


def download_and_probe_all(streams: dict, directory: str, max_workers: int = None):
    """Download and probe all clips across all streams."""
    if max_workers is None:
        # High parallelism for 64-core systems
        max_workers = min(os.cpu_count() or 4, 48)
    
    all_clips = []
    for stream in streams.values():
        all_clips.extend(stream['clips'])

    logging.info(f'Downloading and probing {len(all_clips)} clips with {max_workers} workers...')

    with create_shared_client() as client:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_download_and_probe_clip, clip, directory, client): clip
                for clip in all_clips
            }
            done = 0
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 20 == 0:
                    logging.info(f'  Downloaded/probed {done}/{len(all_clips)} clips')

    logging.info(f'All {len(all_clips)} clips downloaded and probed.')


def _download_slides(slides: List[str], directory: str) -> Dict[str, str]:
    """Download all unique slide images and return URL → local path mapping."""
    url_to_path = {}
    unique_urls = list(dict.fromkeys(slides))  # preserve order, deduplicate

    if not unique_urls:
        return url_to_path

    logging.info(f'Downloading {len(unique_urls)} slide images...')

    with create_shared_client() as client:
        for i, url in enumerate(unique_urls):
            filename = f'slide_{i:03d}.jpg'
            file_path = os.path.join(directory, filename)
            if not os.path.exists(file_path):
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                except Exception as e:
                    logging.warning(f'Failed to download slide {url}: {e}')
                    continue
            url_to_path[url] = file_path

    logging.info(f'Downloaded {len(url_to_path)}/{len(unique_urls)} slide images.')
    return url_to_path


# ─── Step 3: Compute layout timeline ─────────────────────────────────────

def _find_clip_at_time(stream: dict, t: float):
    """Find the clip active at time t in a stream.

    Returns (clip, seek_position) or (None, 0).
    """
    for clip in stream['clips']:
        clip_end = clip['relative_time'] + clip['duration']
        if clip['relative_time'] <= t < clip_end:
            return clip, t - clip['relative_time']
    return None, 0


def compute_layout_timeline(streams: dict, total_duration: float,
                            admin_user_id: Optional[int],
                            hide_silent: bool = False,
                            speech_timelines: Optional[dict] = None,
                            start_time: float = 0,
                            slide_timeline: Optional[List[Tuple[float, float, str]]] = None) -> list:
    """Compute layout segments based on which streams are active at each point.

    Returns list of segments, each with:
        start, end, active_streams: [(stream_key, clip, seek)]
        slide_image: optional local path to a presentation slide image
    """
    if slide_timeline is None:
        slide_timeline = []

    # Collect all change points (clip boundaries)
    change_points = set()
    change_points.add(start_time)

    for stream_key, stream in streams.items():
        for clip in stream['clips']:
            if clip['duration'] <= 0:
                continue
            clip_start = clip['relative_time']
            clip_end = clip_start + clip['duration']
            # Only include points within [start_time, total_duration]
            if clip_end <= start_time or clip_start >= total_duration:
                continue
            if clip_start >= start_time:
                change_points.add(clip_start)
            change_points.add(clip_end)

    # Add slide timeline boundaries as change points
    for sl_start, sl_end, _ in slide_timeline:
        if sl_end <= start_time or sl_start >= total_duration:
            continue
        if sl_start >= start_time:
            change_points.add(sl_start)
        if sl_end <= total_duration:
            change_points.add(sl_end)

    change_points.add(total_duration)

    # Filter out change points that are too close together
    sorted_points = sorted(p for p in change_points if p >= start_time)
    filtered = [sorted_points[0]]
    for t in sorted_points[1:]:
        if t - filtered[-1] >= MIN_SEGMENT_DURATION:
            filtered.append(t)
        elif t == total_duration:
            # Always keep the end
            filtered[-1] = t
    change_points = filtered

    # Build stream priority for camera ordering
    # Admin streams > streams with most total duration > others
    stream_priority = {}
    for stream_key, stream in streams.items():
        if stream['is_screenshare']:
            continue
        total_clip_dur = sum(c['duration'] for c in stream['clips'])
        stream_priority[stream_key] = (
            1 if stream['is_admin'] else 0,
            total_clip_dur,
        )

    # Build segments
    segments = []
    for i in range(len(change_points) - 1):
        t_start = change_points[i]
        t_end = change_points[i + 1]

        if t_start >= total_duration:
            break
        t_end = min(t_end, total_duration)
        duration = t_end - t_start
        if duration < 0.05:
            continue

        t_mid = (t_start + t_end) / 2

        # Find active streams at the segment midpoint
        active = []
        for stream_key, stream in streams.items():
            clip, seek = _find_clip_at_time(stream, t_mid)
            if clip:
                active.append((stream_key, clip, seek))

        # Separate screenshare from cameras
        screenshare = None
        cameras = []
        for stream_key, clip, seek in active:
            if streams[stream_key]['is_screenshare']:
                screenshare = (stream_key, clip, seek)
            else:
                cameras.append((stream_key, clip, seek))

        # Sort cameras by priority
        cameras.sort(key=lambda x: stream_priority.get(x[0], (0, 0)), reverse=True)

        # Filter to only speaking cameras when hide_silent is enabled
        if hide_silent and speech_timelines and cameras:
            speaking = [
                (sk, clip, seek) for sk, clip, seek in cameras
                if _has_speech_at(speech_timelines.get(sk, []), t_mid)
            ]
            # Always keep admin or at least 1 camera
            if not speaking:
                # Keep the first camera (highest priority, likely admin)
                speaking = cameras[:1]
            cameras = speaking

        # Check for file-based presentation slide when no screenshare stream
        slide_image = None
        if screenshare is None and slide_timeline:
            for sl_start, sl_end, sl_path in slide_timeline:
                if sl_start <= t_mid < sl_end:
                    slide_image = sl_path
                    break

        segments.append({
            'start': t_start,
            'end': t_end,
            'screenshare': screenshare,
            'cameras': cameras,
            'slide_image': slide_image,
        })

    if not segments:
        return segments

    # Merge consecutive segments with identical active stream set
    # ONLY if they are contiguous in time to avoid "stretching" clips over gaps
    def _stream_set(seg):
        keys = set()
        if seg['screenshare']:
            sk, clip, _ = seg['screenshare']
            keys.add((sk, clip.get('file_path')))
        for sk, clip, _ in seg['cameras']:
            keys.add((sk, clip.get('file_path')))
        slide = seg.get('slide_image')
        if slide:
            keys.add(('slide', slide))
        return frozenset(keys)

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        # Check for matching stream set AND strict temporal continuity
        if (abs(prev['end'] - seg['start']) < 0.001 and 
            _stream_set(prev) == _stream_set(seg)):
            prev['end'] = seg['end']
        else:
            merged.append(seg)

    logging.info(
        f'Layout timeline: {len(merged)} segments '
        f'(from {len(change_points)} change points, '
        f'duration {total_duration:.0f}s)'
    )
    for i, seg in enumerate(merged):
        sc = 'SCREEN' if seg['screenshare'] else ('SLIDE' if seg.get('slide_image') else 'no_scr')
        cam_v = sum(1 for _, c, _ in seg['cameras'] if c['has_video'])
        cam_a = sum(1 for _, c, _ in seg['cameras'] if c['has_audio'])
        logging.info(
            f'  seg[{i}] {seg["start"]:.1f}-{seg["end"]:.1f}s '
            f'({seg["end"]-seg["start"]:.1f}s) {sc} cams={len(seg["cameras"])}({cam_v}V/{cam_a}A)'
        )
    return merged


# ─── Step 4: Render composite segments ───────────────────────────────────

def _render_segment(seg_index: int, segment: dict, streams: dict,
                    tmpdir: str, threads: int = 1, use_nvenc: bool = False) -> str:
    """Render a single composite segment to a video file.

    Builds an ffmpeg command with filter_complex to overlay video streams
    and mix audio streams.
    """
    output_path = os.path.join(tmpdir, f'seg_{seg_index:05d}.mp4')
    t_start = segment['start']
    duration = segment['end'] - segment['start']

    screenshare = segment['screenshare']  # (stream_key, clip, seek) or None
    cameras = segment['cameras']          # [(stream_key, clip, seek), ...]

    has_screenshare = screenshare is not None

    # Re-resolve clips at segment start time (important for merged segments)
    resolved_screenshare = None
    if screenshare:
        sk = screenshare[0]
        clip, seek = _find_clip_at_time(streams[sk], t_start)
        if clip:
            resolved_screenshare = (sk, clip, seek)
        else:
            logging.warning(
                f'Seg {seg_index}: screenshare stream {sk} has no clip at t={t_start:.1f}s'
            )

    resolved_cameras = []
    for sk, _, _ in cameras:
        clip, seek = _find_clip_at_time(streams[sk], t_start)
        if clip:
            resolved_cameras.append((sk, clip, seek))

    # Use slide image as virtual screenshare when no real screenshare
    slide_image = segment.get('slide_image')
    if not resolved_screenshare and slide_image and os.path.exists(slide_image):
        has_screenshare = True
        logging.debug(f'Seg {seg_index}: using slide image as virtual screenshare')

    mode = 'A(screen+cam)' if has_screenshare else 'B(grid)'
    n_vis = sum(1 for _, c, _ in resolved_cameras if c['has_video'])
    logging.debug(
        f'Seg {seg_index} [{t_start:.1f}-{t_start+duration:.1f}s] mode={mode} '
        f'cams={len(resolved_cameras)}({n_vis}V) screen={resolved_screenshare is not None} '
        f'slide={slide_image is not None}'
    )

    # Determine visible video streams
    if has_screenshare and (resolved_screenshare or slide_image):
        visible_cameras = [
            (sk, clip, seek) for sk, clip, seek in resolved_cameras
            if clip['has_video']
        ][:MAX_SIDEBAR_CAMS]
    else:
        visible_cameras = [
            (sk, clip, seek) for sk, clip, seek in resolved_cameras
            if clip['has_video']
        ][:9]

    inputs = []
    video_roles = []   # (input_idx, role_str)
    audio_indices = [] # input indices that have audio
    input_idx = 0

    # Add screenshare input (real stream or slide image)
    if resolved_screenshare:
        sk, clip, seek = resolved_screenshare
        inputs.extend(['-ss', f'{seek:.3f}', '-t', f'{duration + 1.0:.3f}', '-i', clip['file_path']])
        if clip['has_video']:
            video_roles.append((input_idx, 'main'))
        if clip['has_audio']:
            audio_indices.append(input_idx)
        input_idx += 1
    elif slide_image and has_screenshare:
        # Add slide image as a looped video input (no audio)
        inputs.extend(['-loop', '1', '-framerate', str(OUTPUT_FPS),
                        '-t', f'{duration:.3f}', '-i', slide_image])
        video_roles.append((input_idx, 'main'))
        input_idx += 1

    # Add visible camera inputs
    visible_sks = set()
    for cam_i, (sk, clip, seek) in enumerate(visible_cameras):
        visible_sks.add(sk)
        inputs.extend(['-ss', f'{seek:.3f}', '-t', f'{duration + 1.0:.3f}', '-i', clip['file_path']])
        video_roles.append((input_idx, f'cam_{cam_i}'))
        if clip['has_audio']:
            audio_indices.append(input_idx)
        input_idx += 1

    # Add audio-only camera inputs (limited count)
    MAX_AUDIO_ONLY_INPUTS = 6
    if resolved_screenshare:
        visible_sks.add(resolved_screenshare[0])
    audio_only_added = 0
    for sk, clip, seek in resolved_cameras:
        if sk in visible_sks:
            continue
        if not clip['has_audio']:
            continue
        if not streams[sk].get('conf_has_audio', False):
            continue
        if audio_only_added >= MAX_AUDIO_ONLY_INPUTS:
            break
        inputs.extend(['-ss', f'{seek:.3f}', '-t', f'{duration + 1.0:.3f}', '-i', clip['file_path']])
        audio_indices.append(input_idx)
        input_idx += 1
        audio_only_added += 1

    # No inputs at all → black + silence
    if input_idx == 0:
        _run_ffmpeg([
            '-threads', str(threads),
            '-y',
            '-f', 'lavfi', '-i',
            f'color=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:r={OUTPUT_FPS}:d={duration:.3f}',
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
            '-t', f'{duration:.3f}',
            '-c:v', video_encoder, '-preset', encoder_preset,
            '-c:a', 'aac',
            output_path
        ], desc=f'black seg {seg_index}')
        return output_path

    # Build filter_complex
    filter_parts = []

    # Background canvas
    filter_parts.append(
        f'color=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:r={OUTPUT_FPS}:d={duration:.3f}[bg]'
    )

    current_layer = 'bg'
    overlay_count = 0

    if has_screenshare and (resolved_screenshare or slide_image):
        # ── Mode A: screenshare main + sidebar cameras ──
        main_inputs = [idx for idx, role in video_roles if role == 'main']
        if main_inputs:
            idx = main_inputs[0]
            filter_parts.append(
                f'[{idx}:v]setpts=PTS-STARTPTS'
                f',scale={MAIN_WIDTH}:{MAIN_HEIGHT}'
                f':force_original_aspect_ratio=decrease'
                f',pad={MAIN_WIDTH}:{MAIN_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black'
                f',setsar=1[main_v]'
            )
            next_layer = f'ol{overlay_count}'
            filter_parts.append(
                f'[{current_layer}][main_v]overlay=0:0'
                f':eof_action=repeat[{next_layer}]'
            )
            current_layer = next_layer
            overlay_count += 1

        # Sidebar cameras
        cam_roles = [(idx, role) for idx, role in video_roles
                     if role.startswith('cam_')]
        for idx, role in cam_roles:
            cam_num = int(role.split('_')[1])
            y_pos = cam_num * SIDEBAR_HEIGHT
            label = f'sb{cam_num}'
            filter_parts.append(
                f'[{idx}:v]setpts=PTS-STARTPTS'
                f',scale={SIDEBAR_WIDTH}:{SIDEBAR_HEIGHT}'
                f':force_original_aspect_ratio=decrease'
                f',pad={SIDEBAR_WIDTH}:{SIDEBAR_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black'
                f',setsar=1[{label}]'
            )
            next_layer = f'ol{overlay_count}'
            filter_parts.append(
                f'[{current_layer}][{label}]overlay={MAIN_WIDTH}:{y_pos}'
                f':eof_action=repeat[{next_layer}]'
            )
            current_layer = next_layer
            overlay_count += 1

    else:
        # ── Mode B: grid of cameras ──
        cam_roles = [(idx, role) for idx, role in video_roles
                     if role.startswith('cam_')]
        n_cams = len(cam_roles)
        if n_cams > 0:
            cols, rows = GRID_CONFIGS.get(n_cams, GRID_CONFIGS[min(n_cams, 9)])
            cell_w = OUTPUT_WIDTH // cols
            cell_h = OUTPUT_HEIGHT // rows

            for idx, role in cam_roles:
                cam_num = int(role.split('_')[1])
                col = cam_num % cols
                row = cam_num // cols
                x_pos = col * cell_w
                y_pos = row * cell_h
                label = f'g{cam_num}'
                filter_parts.append(
                    f'[{idx}:v]setpts=PTS-STARTPTS'
                    f',scale={cell_w}:{cell_h}'
                    f':force_original_aspect_ratio=decrease'
                    f',pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:black'
                    f',setsar=1[{label}]'
                )
                next_layer = f'ol{overlay_count}'
                filter_parts.append(
                    f'[{current_layer}][{label}]overlay={x_pos}:{y_pos}'
                    f':eof_action=repeat[{next_layer}]'
                )
                current_layer = next_layer
                overlay_count += 1

    video_out_label = current_layer

    # Audio mixing — use duration=first to avoid blocking when inputs
    # have different lengths; normalize audio from each input first
    if len(audio_indices) > 1:
        # Pad each audio to segment duration with silence to prevent amix stalls
        padded_labels = []
        for ai, idx in enumerate(audio_indices):
            label = f'apad{ai}'
            filter_parts.append(
                f'[{idx}:a]asetpts=PTS-STARTPTS,aresample=44100,apad=whole_dur={duration:.3f}[{label}]'
            )
            padded_labels.append(f'[{label}]')
        amix_in = ''.join(padded_labels)
        filter_parts.append(
            f'{amix_in}amix=inputs={len(audio_indices)}'
            f':duration=first:dropout_transition=0:normalize=1[aout]'
        )
        audio_map = '[aout]'
    elif len(audio_indices) == 1:
        filter_parts.append(
            f'[{audio_indices[0]}:a]asetpts=PTS-STARTPTS,aresample=44100'
            f',apad=whole_dur={duration:.3f}[aout]'
        )
        audio_map = '[aout]'
    else:
        filter_parts.append(f'anullsrc=r=44100:cl=stereo:d={duration:.3f}[aout]')
        audio_map = '[aout]'

    filter_complex = ';'.join(filter_parts)

    # Build complete command
    # Compute a generous timeout: 60s base + 30s per second of output
    ffmpeg_timeout = max(120, int(60 + duration * 30))
    
    video_encoder = 'h264_nvenc' if use_nvenc else 'libx264'
    encoder_preset = 'p1' if use_nvenc else 'ultrafast'  # p1 is fastest for nvenc

    # Add thread_queue_size for each input and use filter_threads
    optimized_inputs = []
    # inputs is a list like ['-ss', '0.0', '-t', '10.0', '-i', 'path', ...]
    i = 0
    while i < len(inputs):
        if inputs[i] == '-i':
            optimized_inputs.extend(['-thread_queue_size', '1024', '-i', inputs[i+1]])
            i += 2
        elif inputs[i] in ('-ss', '-t', '-loop', '-framerate'):
            optimized_inputs.extend([inputs[i], inputs[i+1]])
            i += 2
        else:
            optimized_inputs.append(inputs[i])
            i += 1

    cmd = ['-threads', str(threads), '-filter_threads', str(threads), '-y'] + optimized_inputs
    cmd += ['-filter_complex', filter_complex]
    cmd += ['-map', f'[{video_out_label}]']
    cmd += ['-map', audio_map]
    cmd += [
        '-c:v', video_encoder, '-preset', encoder_preset
    ]
    if not use_nvenc:
        cmd += ['-tune', 'zerolatency']
    
    cmd += [
        '-profile:v', 'high', '-level', '4.1',
        '-pix_fmt', 'yuv420p',
        '-g', '30',
        '-r', str(OUTPUT_FPS),
        '-c:a', 'aac', '-ar', '44100', '-ac', '2',
        '-t', f'{duration:.3f}',
        output_path
    ]

    _run_ffmpeg(cmd, desc=f'segment {seg_index} ({duration:.1f}s)',
                timeout=ffmpeg_timeout)
    return output_path



def render_all_segments(timeline: list, streams: dict, tmpdir: str) -> list:
    """Render all composite segments in parallel."""
    cpu_count = os.cpu_count() or 4
    use_nvenc = _check_nvenc_support()
    
    if use_nvenc:
        # Even with NVENC, we can run more segments (up to 12)
        max_workers = min(cpu_count, 12)
        threads_per_worker = max(1, cpu_count // max_workers)
        logging.info(f'Hardware acceleration (NVENC) detected. Using {max_workers} workers.')
    else:
        # Extreme parallelism for 64-core machines
        if cpu_count >= 60:
            max_workers = 40
        elif cpu_count >= 32:
            max_workers = 24
        elif cpu_count >= 16:
            max_workers = 12
        else:
            max_workers = min(cpu_count, 4)
        
        threads_per_worker = max(1, cpu_count // max_workers)
        logging.info(f'CPU rendering. Using {max_workers} workers with {threads_per_worker} threads each.')

    logging.info(f'Rendering {len(timeline)} segments...')

    segment_files = [None] * len(timeline)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, segment in enumerate(timeline):
            future = executor.submit(
                _render_segment, i, segment, streams, tmpdir, 
                threads=threads_per_worker, use_nvenc=use_nvenc
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            segment_files[idx] = future.result()
            if (idx + 1) % 10 == 0 or idx + 1 == len(timeline):
                logging.info(f'  Rendered segment {idx + 1}/{len(timeline)}')

    return segment_files


# ─── Step 5: Final concat ────────────────────────────────────────────────

def concat_segments(segment_files: list, output_path: str):
    """Concatenate rendered segments into the final video."""
    concat_list_path = segment_files[0].replace(
        os.path.basename(segment_files[0]), 'concat_list.txt'
    )
    with open(concat_list_path, 'w') as f:
        for seg in segment_files:
            f.write(f"file '{seg.replace(chr(92), '/')}'\n")

    _run_ffmpeg([
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-fflags', '+genpts',
        '-i', concat_list_path,
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        '-movflags', '+faststart',
        output_path
    ], desc='final concat')

    logging.info(f'Final composite video saved: {output_path}')


# ─── Main entry point ────────────────────────────────────────────────────

def process_composite_video(directory: str, json_data: dict,
                            output_path: str, max_duration=None,
                            hide_silent: bool = False,
                            start_time: float = 0):
    """Full composite pipeline: parse → download → probe → layout → render → concat."""
    total_duration = float(json_data.get('duration', 0))
    if not total_duration:
        raise ValueError('Duration not found in JSON data.')

    if start_time > 0:
        logging.info(f'Start time offset: {start_time:.0f}s')

    if max_duration:
        end_time = start_time + max_duration
        if end_time < total_duration:
            logging.info(f'Duration limit: {start_time:.0f}s - {end_time:.0f}s (was {total_duration:.0f}s)')
            total_duration = end_time

    # Step 1: Parse event logs into streams
    streams, admin_user_id = parse_event_logs(json_data)

    if not streams:
        logging.error('No media streams found in event logs.')
        return

    # Mark screenshare clips so black-detection is skipped for them,
    # and set VAD flag on camera clips if hide_silent is enabled
    for stream_key, stream in streams.items():
        if stream['is_screenshare']:
            for clip in stream['clips']:
                clip['_is_screenshare'] = True
        elif hide_silent:
            for clip in stream['clips']:
                clip['_run_vad'] = True

    # Step 2: Download and probe all clips
    download_and_probe_all(streams, directory)

    # Step 2b: Reclassify streams by resolution (screenshare detection)
    _reclassify_screenshare_by_dimensions(streams)

    # For reclassified screenshare streams, undo black-detection damage:
    # re-probe clips that were marked has_video=False by blackdetect
    for stream_key, stream in streams.items():
        if not stream['is_screenshare']:
            continue
        for clip in stream['clips']:
            if clip.get('_is_screenshare'):
                continue  # was already screenshare before download
            clip['_is_screenshare'] = True
            # If blackdetect wrongly cleared has_video, re-check via probe
            if not clip['has_video'] and clip.get('file_path'):
                info = _probe_media(clip['file_path'])
                if info['has_video']:
                    clip['has_video'] = True
                    clip['width'] = info['width']
                    clip['height'] = info['height']
                    logging.info(
                        f'Restored has_video for reclassified screenshare clip: '
                        f'{clip["file_path"]}'
                    )

    # Log stream summary
    black_count = 0
    for stream_key, stream in streams.items():
        clips = stream['clips']
        video_clips = sum(1 for c in clips if c['has_video'])
        audio_clips = sum(1 for c in clips if c['has_audio'])
        black_clips = sum(1 for c in clips if not c['has_video'] and not c.get('_is_screenshare'))
        black_count += black_clips
        tag = 'SCREEN' if stream['is_screenshare'] else 'CAM'
        admin_tag = ' [ADMIN]' if stream['is_admin'] else ''
        res_info = ''
        if clips and clips[0]['width']:
            res_info = f' {clips[0]["width"]}x{clips[0]["height"]}'
        logging.info(
            f'  Stream {tag} conf={stream["conference_id"]}: '
            f'{len(clips)} clips ({video_clips}V/{audio_clips}A{res_info}) '
            f'user="{stream["user_name"]}"{admin_tag}'
        )
    if black_count:
        logging.info(f'  Black-screen camera clips removed: {black_count}')

    # Step 2c: Parse file-based presentation timeline
    slides, slide_timeline = parse_presentation_timeline(json_data)
    slide_url_to_path = {}
    if slides:
        slide_url_to_path = _download_slides(slides, directory)
        # Replace URLs with local paths in slide_timeline
        slide_timeline = [
            (start, end, slide_url_to_path[url])
            for start, end, url in slide_timeline
            if url in slide_url_to_path
        ]

    # Step 2d: Build speech timelines if hide_silent is enabled
    speech_timelines = {}
    if hide_silent:
        for stream_key, stream in streams.items():
            if not stream['is_screenshare']:
                speech_timelines[stream_key] = _build_stream_speech_timeline(stream)

    # Step 3: Compute layout timeline
    timeline = compute_layout_timeline(
        streams, total_duration, admin_user_id,
        hide_silent=hide_silent, speech_timelines=speech_timelines,
        start_time=start_time, slide_timeline=slide_timeline
    )

    if not timeline:
        logging.error('No layout segments to render.')
        return

    # Step 4-5: Render segments and concat
    with tempfile.TemporaryDirectory() as tmpdir:
        segment_files = render_all_segments(timeline, streams, tmpdir)
        concat_segments(segment_files, output_path)

    logging.info(f'Composite video complete: {output_path}')


# ─── Legacy entry points (kept for backward compatibility) ───────────────

def process_video_clips(directory: str, json_data: Dict,
                        max_workers: int = None) -> Tuple[float, List[dict], List[dict]]:
    """Legacy: Download clips and classify as video/audio.

    Preserved for backward compatibility. New code should use
    process_composite_video() instead.
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 16)
    
    total_duration = float(json_data.get('duration', 0))
    if not total_duration:
        raise ValueError('Duration not found in JSON data.')

    tasks = []
    for event in json_data.get('eventLogs', []):
        if isinstance(event, dict):
            data = event.get('data', {})
            if isinstance(data, dict) and 'url' in data:
                tasks.append((data['url'], event.get('relativeTime', 0)))

    video_clips = []
    audio_clips = []

    logging.info(f'Downloading {len(tasks)} chunks with {max_workers} parallel workers...')

    with create_shared_client() as client:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_download_and_classify_legacy, url, start_time, directory, client): (url, start_time)
                for url, start_time in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                if result['type'] == 'video':
                    video_clips.append({'path': result['path'], 'start': result['start']})
                else:
                    audio_clips.append({'path': result['path'], 'start': result['start']})

    return total_duration, video_clips, audio_clips


def _download_and_classify_legacy(url, start_time, directory, client):
    downloaded = download_video_chunk(url, directory, client=client)
    try:
        info = _probe_media(downloaded)
        if info['has_video']:
            return {'type': 'video', 'path': downloaded, 'start': start_time}
    except Exception:
        pass
    return {'type': 'audio', 'path': downloaded, 'start': start_time}
