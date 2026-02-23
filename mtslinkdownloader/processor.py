import json
import logging
import math
import os
import re
import sys
import subprocess
import tempfile
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import shutil
import tqdm
import imageio_ffmpeg

from .downloader import download_video_chunk, create_shared_client

# Global stop flag
STOP_REQUESTED = False

# Default Output settings
OUTPUT_FPS = 30
MIN_SEGMENT_DURATION = 0.5
GRID_CONFIGS = {1:(1,1), 2:(2,1), 3:(2,2), 4:(2,2), 5:(3,2), 6:(3,2), 7:(3,3), 8:(3,3), 9:(3,3)}

def request_stop():
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logging.info("!!! Stop requested by user !!!")

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
    ffmpeg = _get_ffmpeg()
    try:
        test_cmd = [ffmpeg, '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1', '-c:v', 'h264_nvenc', '-f', 'null', '-']
        if subprocess.run(test_cmd, capture_output=True, timeout=5).returncode == 0:
            return 'h264_nvenc', 'p1', 2
    except Exception: pass
    try:
        test_cmd = [ffmpeg, '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1', '-c:v', 'h264_qsv', '-f', 'null', '-']
        if subprocess.run(test_cmd, capture_output=True, timeout=5).returncode == 0:
            return 'h264_qsv', 'veryfast', 3
    except Exception: pass
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
    if STOP_REQUESTED: raise RuntimeError('interrupted')
    cmd = [_get_ffmpeg()] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            if result.returncode in (2, 130, -2) or STOP_REQUESTED: raise RuntimeError('interrupted')
            err_msg = result.stderr[-1000:] if result.stderr else "No stderr"
            logging.error(f'FFmpeg fail ({desc}): {err_msg}')
            raise RuntimeError(f'ffmpeg failed: {desc}')
    except subprocess.TimeoutExpired: raise RuntimeError(f'timeout: {desc}')

def _detect_black_video(file_path: str, duration: float) -> bool:
    if duration <= 0 or STOP_REQUESTED: return False
    cmd = [_get_ffmpeg(), '-threads', '2', '-t', str(min(duration, 30.0)), '-i', file_path, '-vf', 'blackdetect=d=0.1:pix_th=0.10', '-an', '-f', 'null', '-']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        total_black = sum(float(m.group(1)) for m in re.finditer(r'black_duration:\s*([\d.]+)', result.stderr))
        return total_black >= min(duration, 30.0) * 0.9
    except Exception: return False

def _detect_speech_intervals(file_path: str, duration: float) -> list:
    if duration <= 0 or STOP_REQUESTED: return []
    cmd = [_get_ffmpeg(), '-threads', '2', '-i', file_path, '-af', 'silencedetect=noise=-40dB:d=0.4', '-vn', '-f', 'null', '-']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        silence = []
        start = None
        for line in result.stderr.splitlines():
            m_s = re.search(r'silence_start:\s*([\d.]+)', line)
            m_e = re.search(r'silence_end:\s*([\d.]+)', line)
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
    if STOP_REQUESTED: return file_path
    proxy_path = file_path.replace('.mp4', '_proxy.mp4')
    if os.path.exists(proxy_path): return proxy_path
    cmd = [_get_ffmpeg(), '-threads', str(threads), '-y', '-i', file_path, '-vf', 'scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2:black,setsar=1', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-c:a', 'copy', proxy_path]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        return proxy_path
    except Exception: return file_path

def _create_name_placeholder(name: str, directory: str, w: int, h: int) -> str:
    safe_name = name.replace("'", "").replace(":", "")
    file_name = f"name_{abs(hash(safe_name))}.png"
    path = os.path.join(directory, file_name)
    if os.path.exists(path): return path
    cmd = [_get_ffmpeg(), '-y', '-f', 'lavfi', '-i', f'color=c=gray:s={w}x{h}:d=1', '-vf', f"drawtext=text='{safe_name}':fontcolor=white:fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2", '-frames:v', '1', path]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return path
    except Exception: return None


def _is_lecturer_stream(stream: dict, admin_user_id: Optional[int]) -> bool:
    user_id = stream.get('user_id')
    if stream.get('is_admin'):
        return True
    return bool(admin_user_id and user_id and user_id == admin_user_id)


def _build_chat_overlay_text(events: Optional[list], t: float, max_lines: int = 6) -> str:
    if not events:
        return "CHAT: no messages yet"
    visible = [e for e in events if e.get('time', 0) <= t]
    if not visible:
        return "CHAT: no messages yet"
    lines = []
    for e in visible[-max_lines:]:
        kind = str(e.get('type', 'CHAT')).upper()
        user = str(e.get('user', 'System')).replace('\n', ' ').strip() or 'System'
        text = str(e.get('text', '')).replace('\n', ' ').strip()
        lines.append(f"[{kind}] {user}: {text}")
    return '\n'.join(lines)


def _escape_drawtext_path(path: str) -> str:
    normalized = path.replace('\\', '/')
    normalized = normalized.replace(':', r'\:')
    normalized = normalized.replace("'", r"\'")
    return normalized

# ─── Parsing ─────────────────────────────────────────────────────────────

def parse_chat_and_questions(json_data: dict) -> Tuple[list, list]:
    event_logs = json_data.get('eventLogs', [])
    chat, questions = [], []
    for event in event_logs:
        module, t, data = event.get('module', ''), event.get('relativeTime', 0), event.get('data', {})
        if module == 'chat.add' and 'text' in data:
            chat.append({'time': t, 'user': data.get('user', {}).get('nickname', 'System'), 'text': data['text'], 'type': 'CHAT'})
        elif module == 'question.add' and 'text' in data:
            questions.append({'time': t, 'user': data.get('user', {}).get('nickname', 'Anonymous'), 'text': data['text'], 'type': 'Q&A'})
    return chat, questions

def format_time_srt(seconds: float) -> str:
    td = math.modf(seconds); ms = int(td[0] * 1000); s = int(td[1]); m, s = divmod(s, 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def export_chat_files(chat: list, questions: list, directory: str, base_name: str):
    if not chat and not questions: return
    txt_path = os.path.join(directory, f"{base_name}_log.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        if chat:
            f.write("=== CHAT MESSAGES ===\n")
            for m in chat: f.write(f"[{int(m['time'])}s] {m['user']}: {m['text']}\n")
        if questions:
            f.write("\n=== QUESTIONS ===\n")
            for q in questions: f.write(f"[{int(q['time'])}s] {q['user']}: {q['text']}\n")
    if chat:
        srt_path = os.path.join(directory, f"{base_name}.srt")
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, m in enumerate(chat):
                start = m['time']; end = start + 5.0
                f.write(f"{i+1}\n{format_time_srt(start)} --> {format_time_srt(end)}\n{m['user']}: {m['text']}\n\n")

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

# ─── Processing Logic ────────────────────────────────────────────────────

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty(): tqdm.tqdm.write(msg)
            else: print(msg, file=sys.stderr)
            self.flush()
        except Exception: self.handleError(record)

def _download_and_probe_clip(clip: dict, directory: str, client) -> Optional[str]:
    if STOP_REQUESTED: return "Interrupted"
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
    global STOP_REQUESTED
    max_workers = max_workers or min(os.cpu_count() or 4, 48)
    all_clips = [c for s in streams.values() for c in s['clips']]
    logging.info(f'Processing {len(all_clips)} clips...')
    reports = []
    tqdm_args = {"total": len(all_clips), "desc": "Processing files", "unit": "file", "ascii": True, "mininterval": 0.5, "leave": True}
    with create_shared_client() as client:
        with tqdm.tqdm(**tqdm_args) as pbar:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures = {executor.submit(_download_and_probe_clip, c, directory, client): c for c in all_clips}
                for f in as_completed(futures):
                    if STOP_REQUESTED: executor.shutdown(wait=False, cancel_futures=True); break
                    res = f.result()
                    if res: reports.append(res)
                    pbar.update(1)
            finally: executor.shutdown(wait=True)
    if STOP_REQUESTED: raise RuntimeError("interrupted")
    for r in reports: logging.info(f'  {r}')

def _find_clip_at_time(stream: dict, t: float):
    for c in stream['clips']:
        if c['relative_time'] <= t < c['relative_time'] + c['duration']:
            return c, t - c['relative_time']
    return None, 0

def compute_layout_timeline(streams: dict, total_duration: float, admin_user_id: Optional[int], hide_silent: bool = False, speech_timelines: Optional[dict] = None, start_time: float = 0, slide_timeline: Optional[list] = None, events: Optional[list] = None) -> list:
    change_points = {start_time, total_duration}
    event_times = sorted(e.get('time', 0) for e in (events or []) if start_time <= e.get('time', 0) <= total_duration)
    for s in streams.values():
        for c in s['clips']:
            if c['duration'] > 0:
                if c['relative_time'] >= start_time: change_points.add(c['relative_time'])
                if c['relative_time'] + c['duration'] <= total_duration: change_points.add(c['relative_time'] + c['duration'])
    for s_s, s_e, _ in (slide_timeline or []):
        if s_s >= start_time: change_points.add(s_s)
        if s_e <= total_duration: change_points.add(s_e)
    for t in event_times:
        change_points.add(t)
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
        all_cameras = [a for a in active if not streams[a[0]]['is_screenshare']]
        audio_sources = [a for a in active if a[1]['has_audio']]
        lecturer = next((c for c in all_cameras if _is_lecturer_stream(streams[c[0]], admin_user_id)), None)
        cameras = list(all_cameras)
        if hide_silent and speech_timelines:
            speaking = [c for c in all_cameras if _has_speech_at(speech_timelines.get(c[0], []), t_m)]
            cameras = ([lecturer] if lecturer else []) + [c for c in speaking if not lecturer or c[0] != lecturer[0]]
        cameras = [c for c in cameras if c[1]['has_video'] or c[1]['has_audio']]
        if lecturer and not any(c[0] == lecturer[0] for c in cameras):
            if lecturer[1]['has_video'] or lecturer[1]['has_audio']:
                cameras = [lecturer] + cameras
        cameras.sort(key=lambda c: (0 if _is_lecturer_stream(streams[c[0]], admin_user_id) else 1, streams[c[0]].get('user_name', '').lower()))
        slide = next((path for s_s, s_e, path in (slide_timeline or []) if s_s <= t_m < s_e), None) if not screenshare else None
        chat_version = bisect_right(event_times, t_m)
        segments.append({'start': t_s, 'end': t_e, 'screenshare': screenshare, 'cameras': cameras, 'audio_sources': audio_sources, 'slide_image': slide, 'chat_version': chat_version})
    if not segments: return []
    merged = [segments[0]]
    for s in segments[1:]:
        prev = merged[-1]
        s_prev = {(sk, c['file_path']) for sk, c, _ in ([prev['screenshare']] if prev['screenshare'] else []) + prev['cameras']}
        if prev.get('slide_image'): s_prev.add(('slide', prev['slide_image']))
        s_prev.add(('chat', prev.get('chat_version', 0)))
        s_curr = {(sk, c['file_path']) for sk, c, _ in ([s['screenshare']] if s['screenshare'] else []) + s['cameras']}
        if s.get('slide_image'): s_curr.add(('slide', s['slide_image']))
        s_curr.add(('chat', s.get('chat_version', 0)))
        if abs(prev['end'] - s['start']) < 0.001 and s_prev == s_curr: prev['end'] = s['end']
        else: merged.append(s)
    return merged

def _render_segment(seg_index: int, segment: dict, streams: dict, tmpdir: str, 
                    threads: int = 1, encoder: str = 'libx264', preset: str = 'ultrafast',
                    out_w: int = 1920, out_h: int = 1080, events: list = None) -> str:
    if STOP_REQUESTED: raise RuntimeError('interrupted')
    output_path = os.path.join(tmpdir, f'seg_{seg_index:05d}.mp4')
    t_start, duration = segment['start'], segment['end'] - segment['start']
    inputs, v_inputs, a_inputs, cur_idx = [], [], [], 0
    main_w = (int(out_w * 0.8333) // 2) * 2; sidebar_w = out_w - main_w; sidebar_h = out_h // 4
    cam_visual_count = 0
    max_cam_slots = 3
    audio_input_by_stream = {}
    added_audio_inputs = set()

    def _append_audio_input(input_idx: int):
        if input_idx in added_audio_inputs:
            return
        if len(a_inputs) >= 8:
            return
        added_audio_inputs.add(input_idx)
        a_inputs.append(input_idx)
    
    if segment['screenshare']:
        sk, _, _ = segment['screenshare']
        clip, seek = _find_clip_at_time(streams[sk], t_start)
        if clip:
            inputs.extend(['-ss', f'{seek:.3f}', '-t', f'{duration+1:.3f}', '-thread_queue_size', '1024', '-i', clip['file_path']])
            if clip['has_video']: v_inputs.append((cur_idx, 'main', None))
            if clip['has_audio']:
                audio_input_by_stream[sk] = cur_idx
                _append_audio_input(cur_idx)
            cur_idx += 1
    elif segment.get('slide_image'):
        inputs.extend(['-loop', '1', '-framerate', str(OUTPUT_FPS), '-t', f'{duration:.3f}', '-i', segment['slide_image']])
        v_inputs.append((cur_idx, 'main', None)); cur_idx += 1

    for sk, _, _ in segment['cameras']:
        clip, seek = _find_clip_at_time(streams[sk], t_start)
        if not clip:
            continue
        user_name = streams[sk].get('user_name', 'User')
        if clip['has_video']:
            path = clip.get('proxy_path', clip['file_path'])
            inputs.extend(['-ss', f'{seek:.3f}', '-t', f'{duration+1:.3f}', '-thread_queue_size', '1024', '-i', path])
            if cam_visual_count < max_cam_slots:
                v_inputs.append((cur_idx, 'cam', user_name))
                cam_visual_count += 1
            if clip['has_audio']:
                audio_input_by_stream[sk] = cur_idx
                _append_audio_input(cur_idx)
            cur_idx += 1
        elif clip['has_audio']:
            img_path = _create_name_placeholder(user_name, os.path.dirname(output_path), sidebar_w, sidebar_h)
            if img_path and cam_visual_count < max_cam_slots:
                inputs.extend(['-loop', '1', '-framerate', str(OUTPUT_FPS), '-t', f'{duration:.3f}', '-i', img_path])
                v_inputs.append((cur_idx, 'cam', user_name))
                cam_visual_count += 1
                cur_idx += 1
            inputs.extend(['-ss', f'{seek:.3f}', '-t', f'{duration+1:.3f}', '-thread_queue_size', '1024', '-i', clip['file_path']])
            audio_input_by_stream[sk] = cur_idx
            _append_audio_input(cur_idx)
            cur_idx += 1

    for sk, _, _ in segment.get('audio_sources', segment['cameras']):
        clip, seek = _find_clip_at_time(streams[sk], t_start)
        if not clip or not clip['has_audio']:
            continue
        if sk in audio_input_by_stream:
            _append_audio_input(audio_input_by_stream[sk])
            continue
        inputs.extend(['-ss', f'{seek:.3f}', '-t', f'{duration+1:.3f}', '-thread_queue_size', '1024', '-i', clip['file_path']])
        audio_input_by_stream[sk] = cur_idx
        _append_audio_input(cur_idx)
        cur_idx += 1

    if not inputs:
        _run_ffmpeg(['-threads', str(threads), '-y', '-f', 'lavfi', '-i', f'color=black:s={out_w}x{out_h}:r={OUTPUT_FPS}:d={duration:.3f}', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo', '-t', f'{duration:.3f}', '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', output_path], desc=f'black {seg_index}')
        return output_path

    filter_parts = [f'color=black:s={out_w}x{out_h}:r={OUTPUT_FPS}:d={duration:.3f}[bg]']
    curr_v, ov_idx, cam_count = 'bg', 0, 0
    for idx, type, name in v_inputs:
        if type == 'main':
            pts = 'setpts=PTS-STARTPTS,' if not (segment.get('slide_image') and not segment['screenshare']) else ''
            filter_parts.append(f'[{idx}:v]{pts}scale={main_w}:{out_h}:force_original_aspect_ratio=decrease,pad=max({main_w}\,iw):max({out_h}\,ih):(ow-iw)/2:(oh-ih)/2:black,setsar=1,scale={main_w}:{out_h}[m_v]')
            filter_parts.append(f'[{curr_v}][m_v]overlay=0:0:eof_action=repeat[v{ov_idx}]')
        elif type == 'main_no_pts':
            filter_parts.append(f'[{idx}:v]scale={sidebar_w}:{sidebar_h},setsar=1[c{idx}]')
            filter_parts.append(f'[{curr_v}][c{idx}]overlay={main_w}:{cam_count*sidebar_h}:eof_action=repeat[v{ov_idx}]')
            cam_count += 1
        else:
            filter_parts.append(f'[{idx}:v]scale={sidebar_w}:{sidebar_h}:force_original_aspect_ratio=decrease,pad=max({sidebar_w}\,iw):max({sidebar_h}\,ih):(ow-iw)/2:(oh-ih)/2:black,setsar=1,scale={sidebar_w}:{sidebar_h}[c{idx}]')
            filter_parts.append(f'[{curr_v}][c{idx}]overlay={main_w}:{cam_count*sidebar_h}:eof_action=repeat[v{ov_idx}]')
            cam_count += 1
        curr_v, ov_idx = f'v{ov_idx}', ov_idx + 1

    chat_txt = _build_chat_overlay_text(events, t_start)
    chat_path = os.path.join(tmpdir, f'chat_{seg_index:05d}.txt')
    with open(chat_path, 'w', encoding='utf-8') as f:
        f.write(chat_txt)
    esc_chat_path = _escape_drawtext_path(chat_path)
    filter_parts.append(
        f"color=black@0.68:s={sidebar_w}x{sidebar_h}:d={duration:.3f},"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.10:t=2,"
        f"drawtext=text='CHAT / Q&A':fontcolor=white:fontsize=16:x=10:y=8,"
        f"drawtext=fontcolor=cyan:fontsize=14:line_spacing=5:x=10:y=34:textfile='{esc_chat_path}':reload=0[chat]"
    )
    filter_parts.append(f'[{curr_v}][chat]overlay={main_w}:{3*sidebar_h}[v_fin]')
    curr_v = 'v_fin'

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

def render_all_segments(timeline: list, streams: dict, tmpdir: str, out_w: int = 1920, out_h: int = 1080, events: list = None) -> list:
    global STOP_REQUESTED
    encoder, preset, max_workers = _detect_best_encoder(); cpu_count = os.cpu_count() or 4
    threads_per_worker = max(2, cpu_count // max_workers); segment_files = [None] * len(timeline)
    tqdm_handler = TqdmLoggingHandler(); root_logger = logging.getLogger()
    tqdm_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S'))
    root_logger.addHandler(tqdm_handler)
    try:
        tqdm_args = {"total": len(timeline), "desc": "Rendering", "unit": "seg", "ascii": True, "mininterval": 0.5, "leave": True}
        with tqdm.tqdm(**tqdm_args) as pbar:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures = {executor.submit(_render_segment, i, s, streams, tmpdir, threads_per_worker, encoder, preset, out_w, out_h, events): i for i, s in enumerate(timeline)}
                for f in as_completed(futures):
                    if STOP_REQUESTED: executor.shutdown(wait=False, cancel_futures=True); break
                    idx = futures[f]
                    try: 
                        res = f.result()
                        if res: segment_files[idx] = res
                        else: raise RuntimeError(f"Seg {idx} fail")
                    except Exception as e:
                        if 'interrupted' not in str(e): logging.error(f'Critical: Segment {idx} failed: {e}')
                        executor.shutdown(wait=False, cancel_futures=True); raise
                    pbar.update(1)
            finally: executor.shutdown(wait=True)
    finally: root_logger.removeHandler(tqdm_handler)
    if STOP_REQUESTED: raise RuntimeError("interrupted")
    return [f for f in segment_files if f]

def concat_segments(segment_files: list, output_path: str):
    if STOP_REQUESTED: return
    list_path = os.path.join(os.path.dirname(segment_files[0]), 'concat.txt')
    with open(list_path, 'w') as f:
        for s in segment_files: f.write(f"file '{s.replace(chr(92), '/')}'\n")
    _run_ffmpeg(['-y', '-f', 'concat', '-safe', '0', '-fflags', '+genpts', '-i', list_path, '-c', 'copy', '-avoid_negative_ts', 'make_zero', '-movflags', '+faststart', output_path], desc='concat')

def process_composite_video(directory: str, json_data: dict, output_path: str, max_duration=None, hide_silent: bool = False, start_time: float = 0, quality: str = "1080p"):
    global STOP_REQUESTED
    STOP_REQUESTED = False 
    res_map = {"1080p": (1920, 1080), "720p": (1280, 720)}
    out_w, out_h = res_map.get(quality, (1920, 1080))
    total_duration = float(json_data.get('duration', 0))
    if max_duration: total_duration = min(total_duration, start_time + max_duration)
    chat, questions = parse_chat_and_questions(json_data)
    all_events = sorted(chat + questions, key=lambda x: x['time'])
    export_chat_files(chat, questions, directory, os.path.basename(output_path).replace('.mp4', ''))
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
                if STOP_REQUESTED: break
                path = os.path.join(directory, f'slide_{i:03d}.jpg')
                if not os.path.exists(path):
                    try:
                        resp = client.get(url); f = open(path, 'wb'); f.write(resp.content); f.close()
                    except Exception: continue
                slide_map[url] = path
        s_timeline = [(s, e, slide_map[u]) for s, e, u in (s_timeline or []) if u in slide_map]
    if STOP_REQUESTED: raise RuntimeError("interrupted")
    speech_timelines = {sk: _build_stream_speech_timeline(s) for sk, s in streams.items() if not s['is_screenshare']} if hide_silent else {}
    timeline = compute_layout_timeline(streams, total_duration, admin_id, hide_silent, speech_timelines, start_time, s_timeline, all_events)
    if not timeline: return
    if hide_silent:
        active = {streams[sk]['user_name'] for sk, t in speech_timelines.items() if t}
        logging.info(f'Active participants (with speech): {len(active)}')
    with tempfile.TemporaryDirectory() as tmp:
        files = render_all_segments(timeline, streams, tmp, out_w, out_h, events=all_events)
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
