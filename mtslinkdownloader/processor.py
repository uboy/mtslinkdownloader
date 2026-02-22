import json
import logging
import math
import os
import re
import sys
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import shutil
import tqdm
import imageio_ffmpeg

from .downloader import download_video_chunk, create_shared_client

# Output dimensions
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 30

# Sidebar layout
MAIN_WIDTH = 1600
MAIN_HEIGHT = 1080
SIDEBAR_WIDTH = 320
SIDEBAR_HEIGHT = 270
MAX_SIDEBAR_CAMS = 4

GRID_CONFIGS = {1:(1,1), 2:(2,1), 3:(2,2), 4:(2,2), 5:(3,2), 6:(3,2), 7:(3,3), 8:(3,3), 9:(3,3)}
MIN_SEGMENT_DURATION = 0.5

# ─── Utilities ──────────────────────────────────────────────────────────

def _get_ffmpeg():
    return shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()

def _get_ffprobe():
    ffmpeg = _get_ffmpeg()
    ffprobe = os.path.join(os.path.dirname(ffmpeg), 'ffprobe')
    for ext in ['', '.exe']:
        if os.path.exists(ffprobe + ext): return ffprobe + ext
    return shutil.which('ffprobe')

def _detect_best_encoder() -> Tuple[str, str, int]:
    """
    Detects the best working hardware encoder.
    Returns: (encoder_name, preset, recommended_workers)
    """
    ffmpeg = _get_ffmpeg()
    # 1. Try NVIDIA NVENC
    try:
        test_cmd = [ffmpeg, '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1', '-c:v', 'h264_nvenc', '-f', 'null', '-']
        if subprocess.run(test_cmd, capture_output=True, timeout=5).returncode == 0:
            return 'h264_nvenc', 'p1', 2 # 2 is safest for consumer GPUs
    except Exception: pass

    # 2. Try Intel QuickSync (QSV)
    try:
        test_cmd = [ffmpeg, '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1', '-c:v', 'h264_qsv', '-f', 'null', '-']
        if subprocess.run(test_cmd, capture_output=True, timeout=5).returncode == 0:
            return 'h264_qsv', 'veryfast', 3 # Intel handles multi-session well
    except Exception: pass

    # 3. Fallback to CPU (libx264)
    cpu_count = os.cpu_count() or 4
    workers = 32 if cpu_count >= 60 else max(1, cpu_count // 2)
    return 'libx264', 'ultrafast', workers

def _probe_media(file_path: str) -> dict:
    cmd = [_get_ffprobe(), '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', file_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
    except Exception: return {'has_video': False, 'has_audio': False, 'width': 0, 'height': 0, 'duration': 0}
    
    info = {'has_video': False, 'has_audio': False, 'width': 0, 'height': 0, 'duration': float(data.get('format', {}).get('duration', 0))}
    for s in data.get('streams', []):
        if s.get('codec_type') == 'video':
            info.update({'has_video': True, 'width': int(s.get('width', 0)), 'height': int(s.get('height', 0))})
        elif s.get('codec_type') == 'audio': info['has_audio'] = True
    return info

def _run_ffmpeg(args: list, desc: str = "", timeout: int = None):
    cmd = [_get_ffmpeg()] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            if result.returncode in (2, 130, -2): raise RuntimeError('interrupted')
            err_msg = result.stderr[-1000:] if result.stderr else "No stderr"
            logging.error(f'FFmpeg fail ({desc}): {err_msg}')
            raise RuntimeError(f'ffmpeg failed: {desc}')
    except subprocess.TimeoutExpired: raise RuntimeError(f'timeout: {desc}')

def _detect_black_video(file_path: str, duration: float) -> bool:
    if duration <= 0: return False
    cmd = [_get_ffmpeg(), '-threads', '2', '-t', str(min(duration, 30.0)), '-i', file_path, '-vf', 'blackdetect=d=0.1:pix_th=0.10', '-an', '-f', 'null', '-']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        total_black = sum(float(m.group(1)) for m in re.finditer(r'black_duration:\s*([\d.]+)', result.stderr))
        return total_black >= min(duration, 30.0) * 0.9
    except Exception: return False

def _detect_speech_intervals(file_path: str, duration: float) -> list:
    if duration <= 0: return []
    cmd = [_get_ffmpeg(), '-threads', '2', '-i', file_path, '-af', 'silencedetect=noise=-35dB:d=0.5', '-vn', '-f', 'null', '-']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        silence = []; start = None
        for line in result.stderr.splitlines():
            m_s, m_e = re.search(r'silence_start:\s*([\d.]+)', line), re.search(r'silence_end:\s*([\d.]+)', line)
            if m_s: start = float(m_s.group(1))
            if m_e and start is not None: silence.append((start, float(m_e.group(1)))); start = None
        if start is not None: silence.append((start, duration))
        speech = []; prev = 0.0
        for s_s, s_e in sorted(silence):
            if s_s > prev: speech.append((prev, s_s))
            prev = max(prev, s_e)
        if prev < duration: speech.append((prev, duration))
        return speech
    except Exception: return [(0, duration)]

def _build_stream_speech_timeline(stream: dict) -> list:
    intervals = []
    for c in stream['clips']:
        for s, e in c.get('speech_intervals', []):
            intervals.append((c['relative_time'] + s, c['relative_time'] + e))
    if not intervals: return []
    intervals.sort(); merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]: merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else: merged.append((s, e))
    return merged

def _has_speech_at(timeline: list, t: float) -> bool:
    for s, e in timeline:
        if s <= t <= e: return True
        if s > t: break
    return False

def _create_proxy_clip(file_path: str, threads: int = 2) -> str:
    proxy_path = file_path.replace('.mp4', '_proxy.mp4')
    if os.path.exists(proxy_path): return proxy_path
    cmd = [_get_ffmpeg(), '-threads', str(threads), '-y', '-i', file_path, '-vf', 'scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2:black,setsar=1', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-c:a', 'copy', proxy_path]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        return proxy_path
    except Exception: return file_path

# ─── Parsing ─────────────────────────────────────────────────────────────

def parse_event_logs(json_data: dict) -> Tuple[dict, Optional[int]]:
    event_logs = json_data.get('eventLogs', [])
    admin_user_id = None
    for event in event_logs:
        if event.get('module') == 'eventsession.start':
            snapshot = event.get('snapshot', {}).get('data', {})
            for user_entry in snapshot.get('userlist', []):
                if user_entry.get('role') == 'ADMIN':
                    admin_user_id = user_entry.get('user', {}).get('id'); break
            break
    conf_meta = {}
    for event in event_logs:
        if event.get('module') == 'conference.add':
            d = event.get('data', {}); conf_id = d.get('id')
            if conf_id:
                user = d.get('user', {})
                conf_meta[conf_id] = {'has_video': d.get('hasVideo', False), 'has_audio': d.get('hasAudio', False), 'user_id': user.get('id'), 'user_name': user.get('nickname', ''), 'participation_id': d.get('participationId')}
    streams = {}
    for event in event_logs:
        if event.get('module') != 'mediasession.add': continue
        data = event.get('data', {})
        if not isinstance(data, dict) or 'url' not in data: continue
        stream_data = data.get('stream', {}); rel_time = event.get('relativeTime', 0); url = data['url']
        is_screenshare = 'screensharing' in stream_data
        if not is_screenshare:
            s_type, s_name = str(stream_data.get('type', '')).lower(), str(stream_data.get('name', '')).lower()
            for k in ('screen', 'presentation', 'desktop'):
                if k in s_type or k in s_name: is_screenshare = True; break
        if is_screenshare:
            s_info = stream_data.get('screensharing', {}); conf_id = s_info.get('id') or stream_data.get('conference', {}).get('id')
            sk = (conf_id, True)
        else: sk = (stream_data.get('conference', {}).get('id'), False)
        if sk not in streams:
            meta = conf_meta.get(sk[0], {}); user_id = meta.get('user_id')
            streams[sk] = {'conference_id': sk[0], 'is_screenshare': is_screenshare, 'user_id': user_id, 'user_name': meta.get('user_name', ''), 'conf_has_video': meta.get('has_video', False), 'conf_has_audio': meta.get('has_audio', False), 'is_admin': user_id == admin_user_id if user_id and admin_user_id else False, 'clips': []}
        streams[sk]['clips'].append({'url': url, 'relative_time': rel_time, 'file_path': None, 'duration': 0, 'width': 0, 'height': 0, 'has_video': False, 'has_audio': False})
    for s in streams.values(): s['clips'].sort(key=lambda c: c['relative_time'])
    return streams, admin_user_id

def parse_presentation_timeline(json_data: dict) -> Tuple[List[str], List[Tuple[float, float, str]]]:
    event_logs = json_data.get('eventLogs', [])
    pres_events = [e for e in event_logs if e.get('module') == 'presentation.update']
    slides, slide_timeline = [], []
    if not pres_events: return slides, slide_timeline
    for event in pres_events:
        slide_list = event.get('data', {}).get('fileReference', {}).get('file', {}).get('slides', [])
        if slide_list: slides = [s.get('url', '') for s in slide_list if s.get('url')]; break
    if not slides: return slides, slide_timeline
    idx, start, url = 0, None, None
    for event in pres_events:
        d = event.get('data', {}); t = event.get('relativeTime', 0)
        if d.get('isActive', False):
            if start is not None: slide_timeline.append((start, t, url))
            url = slides[idx % len(slides)]; start = t; idx += 1
        elif start is not None:
            slide_timeline.append((start, t, url)); start = None; url = None
    if start is not None: slide_timeline.append((start, float(json_data.get('duration', 0)), url))
    return slides, slide_timeline

def _reclassify_screenshare_by_dimensions(streams: dict):
    reclass = []
    for sk, s in streams.items():
        if s['is_screenshare']: continue
        v_clips = [c for c in s['clips'] if c['has_video'] and c['width'] > 0]
        if not v_clips: continue
        hd = sum(1 for c in v_clips if c['width'] >= 1280 and c['height'] >= 720)
        if hd > len(v_clips) / 2: reclass.append((sk, (s['conference_id'], True)))
    for old, new in reclass:
        if new not in streams:
            s = streams[old]; s['is_screenshare'] = True; streams[new] = s; del streams[old]

# ─── Processing ──────────────────────────────────────────────────────────

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if sys.stderr.isatty(): tqdm.tqdm.write(msg)
            else: print(msg, file=sys.stderr)
            self.flush()
        except Exception: self.handleError(record)

def _download_and_probe_clip(clip: dict, directory: str, client) -> Optional[str]:
    try:
        file_path = download_video_chunk(clip['url'], directory, client=client)
        clip['file_path'] = file_path
    except Exception as e: return f'Fail: {clip["url"]} ({e})'
    info = _probe_media(file_path)
    clip.update({'duration': info['duration'], 'width': info['width'], 'height': info['height'], 'has_video': info['has_video'], 'has_audio': info['has_audio']})
    if clip['has_video'] and not clip.get('_is_screenshare'):
        clip['proxy_path'] = _create_proxy_clip(file_path)
        if clip['duration'] > 0 and _detect_black_video(file_path, clip['duration']):
            clip['has_video'] = False; return f'Black video: {os.path.basename(file_path)}'
    if clip.get('_run_vad') and clip['has_audio'] and clip['duration'] > 0:
        clip['speech_intervals'] = _detect_speech_intervals(file_path, clip['duration'])
    return None

def download_and_probe_all(streams: dict, directory: str, max_workers: int = None):
    max_workers = max_workers or min(os.cpu_count() or 4, 48)
    all_clips = [c for s in streams.values() for c in s['clips']]
    logging.info(f'Processing {len(all_clips)} clips...')
    reports = []
    tqdm_args = {"total": len(all_clips), "desc": "Processing files", "unit": "file", "ascii": True, "mininterval": 0.5}
    with create_shared_client() as client:
        with tqdm.tqdm(**tqdm_args) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_download_and_probe_clip, c, directory, client): c for c in all_clips}
                for f in as_completed(futures):
                    res = f.result(); 
                    if res: reports.append(res)
                    pbar.update(1)
    for r in reports: logging.info(f'  {r}')

def _find_clip_at_time(stream: dict, t: float):
    for c in stream['clips']:
        if c['relative_time'] <= t < c['relative_time'] + c['duration']:
            return c, t - c['relative_time']
    return None, 0

def compute_layout_timeline(streams: dict, total_duration: float, admin_user_id: Optional[int], hide_silent: bool = False, speech_timelines: Optional[dict] = None, start_time: float = 0, slide_timeline: Optional[list] = None) -> list:
    change_points = {start_time, total_duration}
    for s in streams.values():
        for c in s['clips']:
            if c['duration'] > 0:
                if c['relative_time'] >= start_time: change_points.add(c['relative_time'])
                if c['relative_time'] + c['duration'] <= total_duration: change_points.add(c['relative_time'] + c['duration'])
    for s_s, s_e, _ in (slide_timeline or []):
        if s_s >= start_time: change_points.add(s_s)
        if s_e <= total_duration: change_points.add(s_e)
    sorted_p = sorted(list(change_points))
    filtered = [sorted_p[0]]
    for p in sorted_p[1:]:
        if p - filtered[-1] >= MIN_SEGMENT_DURATION or p == total_duration: filtered.append(p)
    segments = []
    for i in range(len(filtered)-1):
        t_s, t_e = filtered[i], filtered[i+1]; t_m = (t_s + t_e) / 2; active = []
        for sk, s in streams.items():
            clip, seek = _find_clip_at_time(s, t_m)
            if clip: active.append((sk, clip, seek))
        screenshare = next((a for a in active if streams[a[0]]['is_screenshare']), None)
        cameras = [a for a in active if not streams[a[0]]['is_screenshare']]
        if hide_silent and speech_timelines and cameras:
            speaking = [c for c in cameras if _has_speech_at(speech_timelines.get(c[0], []), t_m)]
            cameras = speaking if speaking else cameras[:1]
        cameras = [c for c in cameras if c[1]['has_video'] or c[1]['has_audio']]
        slide = next((path for s_s, s_e, path in (slide_timeline or []) if s_s <= t_m < s_e), None) if not screenshare else None
        segments.append({'start': t_s, 'end': t_e, 'screenshare': screenshare, 'cameras': cameras, 'slide_image': slide})
    if not segments: return []
    merged = [segments[0]]
    for s in segments[1:]:
        prev = merged[-1]
        s_prev = {(sk, c['file_path']) for sk, c, _ in ([prev['screenshare']] if prev['screenshare'] else []) + prev['cameras']}
        if prev.get('slide_image'): s_prev.add(('slide', prev['slide_image']))
        s_curr = {(sk, c['file_path']) for sk, c, _ in ([s['screenshare']] if s['screenshare'] else []) + s['cameras']}
        if s.get('slide_image'): s_curr.add(('slide', s['slide_image']))
        if abs(prev['end'] - s['start']) < 0.001 and s_prev == s_curr: prev['end'] = s['end']
        else: merged.append(s)
    return merged

def _render_segment(seg_index: int, segment: dict, streams: dict, tmpdir: str, threads: int = 1, encoder: str = 'libx264', preset: str = 'ultrafast') -> str:
    output_path = os.path.join(tmpdir, f'seg_{seg_index:05d}.mp4')
    duration = segment['end'] - segment['start']
    inputs, v_inputs, a_inputs, cur_idx = [], [], [], 0
    if segment['screenshare']:
        sk, _, _ = segment['screenshare']; clip, seek = _find_clip_at_time(streams[sk], segment['start'])
        if clip:
            inputs.extend(['-ss', f'{seek:.3f}', '-t', f'{duration+1:.3f}', '-thread_queue_size', '1024', '-i', clip['file_path']])
            if clip['has_video']: v_inputs.append((cur_idx, 'main'))
            if clip['has_audio']: a_inputs.append(cur_idx)
            cur_idx += 1
    elif segment.get('slide_image'):
        inputs.extend(['-loop', '1', '-framerate', str(OUTPUT_FPS), '-t', f'{duration:.3f}', '-i', segment['slide_image']])
        v_inputs.append((cur_idx, 'main')); cur_idx += 1
    for sk, _, _ in segment['cameras']:
        clip, seek = _find_clip_at_time(streams[sk], segment['start'])
        if clip and (clip['has_video'] or clip['has_audio']):
            path = clip.get('proxy_path', clip['file_path'])
            inputs.extend(['-ss', f'{seek:.3f}', '-t', f'{duration+1:.3f}', '-thread_queue_size', '1024', '-i', path])
            if clip['has_video'] and len(v_inputs) < 5: v_inputs.append((cur_idx, 'cam'))
            if clip['has_audio'] and len(a_inputs) < 8: a_inputs.append(cur_idx)
            cur_idx += 1
    if not inputs:
        _run_ffmpeg(['-threads', str(threads), '-y', '-f', 'lavfi', '-i', f'color=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:r={OUTPUT_FPS}:d={duration:.3f}', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo', '-t', f'{duration:.3f}', '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', output_path], desc=f'black {seg_index}')
        return output_path
    filter_parts = [f'color=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:r={OUTPUT_FPS}:d={duration:.3f}[bg]']
    curr_v, ov_idx, cam_count = 'bg', 0, 0
    for idx, type in v_inputs:
        if type == 'main':
            pts = 'setpts=PTS-STARTPTS,' if not (segment.get('slide_image') and not segment['screenshare']) else ''
            filter_parts.append(f'[{idx}:v]{pts}scale={MAIN_WIDTH}:{MAIN_HEIGHT}:force_original_aspect_ratio=decrease,pad={MAIN_WIDTH}:{MAIN_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[m_v]')
            filter_parts.append(f'[{curr_v}][m_v]overlay=0:0:eof_action=repeat[v{ov_idx}]')
        else:
            filter_parts.append(f'[{curr_v}][{idx}:v]overlay={MAIN_WIDTH}:{cam_count*SIDEBAR_HEIGHT}:eof_action=repeat[v{ov_idx}]')
            cam_count += 1
        curr_v, ov_idx = f'v{ov_idx}', ov_idx + 1
    if a_inputs:
        for i, idx in enumerate(a_inputs): filter_parts.append(f'[{idx}:a]asetpts=PTS-STARTPTS,aresample=44100,apad=whole_dur={duration:.3f}[a{i}]')
        filter_parts.append(f'{"".join(f"[a{i}]" for i in range(len(a_inputs)))}amix=inputs={len(a_inputs)}:duration=first:dropout_transition=0:normalize=1[aout]')
        a_map = '[aout]'
    else: filter_parts.append(f'anullsrc=r=44100:cl=stereo:d={duration:.3f}[aout]'); a_map = '[aout]'
    is_min = not segment['screenshare'] and not a_inputs
    cmd = ['-threads', str(threads), '-filter_threads', str(threads), '-y'] + inputs
    cmd += ['-filter_complex', ';'.join(filter_parts), '-map', f'[{curr_v}]', '-map', a_map, '-c:v', encoder, '-preset', preset]
    if is_min and encoder == 'libx264': cmd += ['-crf', '45']
    if encoder == 'libx264': cmd += ['-tune', 'stillimage' if is_min else 'zerolatency']
    cmd += ['-pix_fmt', 'yuv420p', '-r', str(OUTPUT_FPS), '-c:a', 'aac', '-ar', '44100', '-ac', '2', '-t', f'{duration:.3f}', output_path]
    _run_ffmpeg(cmd, desc=f'seg {seg_index}', timeout=max(120, int(60+duration*30)))
    return output_path

def render_all_segments(timeline: list, streams: dict, tmpdir: str) -> list:
    encoder, preset, max_workers = _detect_best_encoder()
    cpu_count = os.cpu_count() or 4
    threads_per_worker = max(2, cpu_count // max_workers)
    logging.info(f'Encoder: {encoder}, Workers: {max_workers}, Threads/Worker: {threads_per_worker}')
    segment_files = [None] * len(timeline)
    tqdm_handler = TqdmLoggingHandler(); root_logger = logging.getLogger()
    tqdm_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S'))
    root_logger.addHandler(tqdm_handler)
    try:
        tqdm_args = {"total": len(timeline), "desc": "Rendering", "unit": "seg", "ascii": True, "mininterval": 0.5}
        with tqdm.tqdm(**tqdm_args) as pbar:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures = {executor.submit(_render_segment, i, s, streams, tmpdir, threads_per_worker, encoder, preset): i for i, s in enumerate(timeline)}
                for f in as_completed(futures):
                    idx = futures[f]
                    try: 
                        res = f.result()
                        if res: segment_files[idx] = res
                        else: raise RuntimeError(f"Seg {idx} fail")
                    except Exception as e:
                        if 'interrupted' not in str(e): logging.error(f'Critical: Segment {idx} failed: {e}')
                        executor.shutdown(wait=False, cancel_futures=True); raise
                    pbar.update(1)
            except KeyboardInterrupt: executor.shutdown(wait=False, cancel_futures=True); raise
            finally: executor.shutdown(wait=True)
    finally: root_logger.removeHandler(tqdm_handler)
    return segment_files

def concat_segments(segment_files: list, output_path: str):
    list_path = os.path.join(os.path.dirname(segment_files[0]), 'concat.txt')
    with open(list_path, 'w') as f:
        for s in segment_files: f.write(f"file '{s.replace(chr(92), '/')}'\n")
    _run_ffmpeg(['-y', '-f', 'concat', '-safe', '0', '-fflags', '+genpts', '-i', list_path, '-c', 'copy', '-avoid_negative_ts', 'make_zero', '-movflags', '+faststart', output_path], desc='concat')

def process_composite_video(directory: str, json_data: dict, output_path: str, max_duration=None, hide_silent: bool = False, start_time: float = 0):
    total_duration = float(json_data.get('duration', 0))
    if max_duration: total_duration = min(total_duration, start_time + max_duration)
    streams, admin_id = parse_event_logs(json_data)
    if not streams: return
    for s in streams.values():
        for c in s['clips']:
            if s['is_screenshare']: c['_is_screenshare'] = True
            elif hide_silent: c['_run_vad'] = True
    download_and_probe_all(streams, directory)
    _reclassify_screenshare_by_dimensions(streams)
    slides, s_timeline = parse_presentation_timeline(json_data)
    if slides:
        slide_map = {}
        with create_shared_client() as client:
            for i, url in enumerate(list(dict.fromkeys(slides))):
                path = os.path.join(directory, f'slide_{i:03d}.jpg')
                if not os.path.exists(path):
                    try:
                        resp = client.get(url); f = open(path, 'wb'); f.write(resp.content); f.close()
                    except Exception: continue
                slide_map[url] = path
        s_timeline = [(s, e, slide_map[u]) for s, e, u in (s_timeline or []) if u in slide_map]
    speech_timelines = {sk: _build_stream_speech_timeline(s) for sk, s in streams.items() if not s['is_screenshare']} if hide_silent else {}
    timeline = compute_layout_timeline(streams, total_duration, admin_id, hide_silent, speech_timelines, start_time, s_timeline)
    if not timeline: return
    if hide_silent:
        active = {streams[sk]['user_name'] for sk, t in speech_timelines.items() if t}
        logging.info(f'Active participants: {len(active)} ({", ".join(list(active)[:5])}...)')
    with tempfile.TemporaryDirectory() as tmp:
        files = render_all_segments(timeline, streams, tmp)
        if files: concat_segments(files, output_path)
    logging.info(f'Done: {output_path}')

def process_video_clips(directory: str, json_data: Dict, max_workers: int = 16) -> Tuple[float, List[dict], List[dict]]:
    total_duration = float(json_data.get('duration', 0))
    tasks = [(e['data']['url'], e.get('relativeTime', 0)) for e in json_data.get('eventLogs', []) if isinstance(e, dict) and 'url' in e.get('data', {})]
    video, audio = [], []
    with create_shared_client() as client:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_download_and_classify_legacy, url, t, directory, client) for url, t in tasks]
            for f in as_completed(futures):
                res = f.result()
                if res['type'] == 'video': video.append({'path': res['path'], 'start': res['start']})
                else: audio.append({'path': res['path'], 'start': res['start']})
    return total_duration, video, audio

def _download_and_classify_legacy(url, start_time, directory, client):
    path = download_video_chunk(url, directory, client=client)
    try:
        if _probe_media(path)['has_video']: return {'type': 'video', 'path': path, 'start': start_time}
    except Exception: pass
    return {'type': 'audio', 'path': path, 'start': start_time}
