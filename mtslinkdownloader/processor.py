import json
import logging
import math
import os
import re
import sys
import subprocess
import tempfile
import datetime
import time
import textwrap
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple, Union

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
    # Conservative parallelism for software x264: too many workers often slows down total render time.
    workers = max(1, min(8, cpu_count // 4))
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

    def _was_interrupted(returncode: int, stderr_text: str) -> bool:
        if returncode in (2, 130, -2):
            return True
        text = (stderr_text or '').lower()
        interrupt_markers = (
            'received signal 2',
            'immediate exit requested',
            'exiting normally, received signal',
            'signal 2',
            'signal 15',
        )
        return any(marker in text for marker in interrupt_markers)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            if STOP_REQUESTED or _was_interrupted(result.returncode, result.stderr):
                raise RuntimeError('interrupted')
            err_msg = result.stderr[-1000:] if result.stderr else "No stderr"
            logging.error(f'FFmpeg fail ({desc}): {err_msg}')
            raise RuntimeError(f'ffmpeg failed: {desc}')
    except KeyboardInterrupt:
        request_stop()
        raise RuntimeError('interrupted')
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


def _choose_primary_screenshare(active: list, streams: dict, admin_user_id: Optional[int]) -> Optional[tuple]:
    screenshares = [a for a in active if streams[a[0]]['is_screenshare']]
    if not screenshares:
        return None

    def _score(item: tuple) -> tuple:
        sk, clip, _ = item
        stream = streams[sk]
        width = int(clip.get('width') or 0)
        height = int(clip.get('height') or 0)
        area = width * height
        clip_start = float(clip.get('relative_time') or 0.0)
        return (
            1 if clip.get('has_video') else 0,
            area,
            1 if clip.get('has_audio') else 0,
            clip_start,
            1 if _is_lecturer_stream(stream, admin_user_id) else 0,
            str(sk),
        )

    return max(screenshares, key=_score)


def _speech_overlap_duration(timeline: list, start: float, end: float) -> float:
    total = 0.0
    for s, e in timeline:
        if s >= end:
            break
        overlap = min(e, end) - max(s, start)
        if overlap > 0:
            total += overlap
    return total

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


def _format_hhmmss(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_chat_overlay_text(events: Optional[list], t: float, max_lines: int = 8, wrap_chars: int = 40) -> str:
    if not events:
        return "[CHAT 00:00:00]\nno messages yet"
    visible = [e for e in events if e.get('time', 0) <= t]
    if not visible:
        return "[CHAT 00:00:00]\nno messages yet"
    blocks = []
    for e in visible:
        kind = str(e.get('type', 'CHAT')).upper()
        stamp = _format_hhmmss(float(e.get('time', 0)))
        user = str(e.get('user', 'System')).replace('\n', ' ').strip() or 'System'
        header = f"[{kind} {stamp}] {user}"
        text = str(e.get('text', '')).replace('\n', ' ').strip()
        wrapped_body = textwrap.wrap(text, width=max(16, wrap_chars), break_long_words=True, break_on_hyphens=False)
        if not wrapped_body:
            wrapped_body = ["(empty)"]
        blocks.append([header] + wrapped_body)

    selected_lines = []
    lines_left = max(2, max_lines)
    for block in reversed(blocks):
        need = len(block)
        if need <= lines_left:
            selected_lines = block + selected_lines
            lines_left -= need
            continue
        if not selected_lines:
            selected_lines = block[:lines_left]
            lines_left = 0
        break
    return '\n'.join(selected_lines)


def _escape_drawtext_path(path: str) -> str:
    normalized = path.replace('\\', '/')
    normalized = normalized.replace(':', r'\:')
    normalized = normalized.replace("'", r"\'")
    return normalized


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _write_status_file(status_file: Optional[str], payload: dict):
    if not status_file:
        return
    try:
        directory = os.path.dirname(status_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = status_file + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, status_file)
    except Exception as exc:
        logging.warning('Failed to write status file %s: %s', status_file, exc)


def _extract_text_value(data: dict) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    candidates = [
        data.get('text'),
        data.get('message'),
        data.get('content'),
        data.get('body'),
        data.get('htmlText'),
        data.get('plainText'),
    ]
    nested = [
        data.get('payload'),
        data.get('message'),
        data.get('chat'),
        data.get('question'),
        data.get('comment'),
        data.get('value'),
    ]
    for n in nested:
        if isinstance(n, dict):
            candidates.extend([
                n.get('text'),
                n.get('message'),
                n.get('content'),
                n.get('body'),
                n.get('htmlText'),
                n.get('plainText'),
            ])
    for c in candidates:
        if isinstance(c, str):
            text = c.strip()
            if text:
                return text
    return None


def _extract_user_name(data: dict, default_name: str) -> str:
    if not isinstance(data, dict):
        return default_name
    candidates = [
        data.get('nickname'),
        data.get('userName'),
        data.get('authorName'),
    ]
    for key in ('user', 'sender', 'author', 'from'):
        v = data.get(key)
        if isinstance(v, dict):
            candidates.extend([v.get('nickname'), v.get('name'), v.get('displayName')])
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return default_name


def _safe_int(value) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None


def _extract_slide_url(data: dict, slides: List[str], current_url: Optional[str], deck_slides: Optional[List[str]] = None) -> Optional[str]:
    if not isinstance(data, dict):
        return current_url

    direct_urls = []
    # Check nested nodes first as they are more specific to the current slide
    for node_key in ('slide', 'currentSlide', 'presentation', 'page'):
        node = data.get(node_key)
        if isinstance(node, dict):
            for key in ('url', 'slideUrl', 'imageUrl'):
                v = node.get(key)
                if isinstance(v, str) and v.strip():
                    if any(v.lower().endswith(ext) for ext in ('.pptx', '.pdf', '.ppt', '.doc', '.docx', '.xls', '.xlsx')):
                        continue
                    direct_urls.append(v.strip())
    
    # Root level URLs as fallback
    for key in ('url', 'slideUrl', 'currentSlideUrl', 'imageUrl'):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            if any(v.lower().endswith(ext) for ext in ('.pptx', '.pdf', '.ppt', '.doc', '.docx', '.xls', '.xlsx')):
                continue
            direct_urls.append(v.strip())
    
    ref_url = data.get('fileReference', {}).get('file', {}).get('currentSlide', {}).get('url')
    if isinstance(ref_url, str) and ref_url.strip():
        if not any(ref_url.lower().endswith(ext) for ext in ('.pptx', '.pdf', '.ppt', '.doc', '.docx', '.xls', '.xlsx')):
            direct_urls.append(ref_url.strip())
    for u in direct_urls:
        return u

    idx_candidates = []
    for key in ('slideIndex', 'currentSlideIndex', 'index', 'pageIndex', 'slideNumber', 'page'):
        idx = _safe_int(data.get(key))
        if idx is not None:
            idx_candidates.append(idx)
    for node_key in ('slide', 'currentSlide', 'presentation', 'page'):
        node = data.get(node_key)
        if isinstance(node, dict):
            for key in ('index', 'slideIndex', 'currentSlideIndex', 'number', 'page', 'slideNumber'):
                idx = _safe_int(node.get(key))
                if idx is not None:
                    idx_candidates.append(idx)

    lookup_sources = []
    if deck_slides:
        lookup_sources.append(deck_slides)
    if slides:
        lookup_sources.append(slides)

    for idx in idx_candidates:
        for source in lookup_sources:
            # Try 0-based first, then 1-based.
            # Most MTS Link events use 0-based slideIndex, but some might use 1-based 'page' or 'number'.
            if 0 <= idx < len(source):
                # If we have exactly one slide, or if idx is small enough, it's ambiguous.
                # But usually 0-based is preferred for 'slideIndex'.
                return source[idx]
            if 1 <= idx <= len(source):
                return source[idx - 1]

    if deck_slides and len(deck_slides) == 1 and deck_slides[0]:
        return deck_slides[0]

    if current_url:
        return current_url
    if slides:
        return slides[0]
    return None

# ─── Parsing ─────────────────────────────────────────────────────────────

def parse_chat_and_questions(json_data: dict) -> Tuple[list, list]:
    event_logs = json_data.get('eventLogs', [])
    chat, questions = [], []
    for event in event_logs:
        module, t, data = str(event.get('module', '')).lower(), event.get('relativeTime', 0), event.get('data', {})
        if not isinstance(data, dict):
            continue
        text = _extract_text_value(data)
        if not text:
            continue
        event_type = str(data.get('type', '')).lower()
        is_question = 'question' in module or event_type in {'question', 'q&a', 'qa'}
        is_chat = ('chat' in module) or ('message' in module) or ('comment' in module) or event_type == 'chat'
        if is_question:
            questions.append({'time': t, 'user': _extract_user_name(data, 'Anonymous'), 'text': text, 'type': 'Q&A'})
        elif is_chat:
            chat.append({'time': t, 'user': _extract_user_name(data, 'System'), 'text': text, 'type': 'CHAT'})
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
            if conf_id is not None and not (isinstance(conf_id, str) and not conf_id.strip()):
                user = d.get('user', {})
                conf_meta[conf_id] = {'has_video': d.get('hasVideo', False), 'has_audio': d.get('hasAudio', False), 'user_id': user.get('id'), 'user_name': user.get('nickname', ''), 'participation_id': d.get('participationId')}
    
    streams = {}
    stream_id_to_sk = {}  # stream.id -> sk

    for event in event_logs:
        module = event.get('module', '')
        rel_time = event.get('relativeTime', 0)
        data = event.get('data', {})
        if not isinstance(data, dict): continue

        if module == 'mediasession.add':
            if 'url' not in data: continue
            stream_data = data.get('stream', {}); url = data['url']
            stream_id = stream_data.get('id')
            is_screenshare = 'screensharing' in stream_data
            if not is_screenshare:
                s_type, s_name = str(stream_data.get('type', '')).lower(), str(stream_data.get('name', '')).lower()
                for k in ('screen', 'presentation', 'desktop'):
                    if k in s_type or k in s_name: is_screenshare = True; break
            
            conference = stream_data.get('conference', {})
            if not isinstance(conference, dict): conference = {}
            s_info = stream_data.get('screensharing', {}) if is_screenshare else {}
            if not isinstance(s_info, dict): s_info = {}
            
            if is_screenshare:
                conf_id = s_info.get('id') or conference.get('id')
            else:
                conf_id = conference.get('id')
            
            if conf_id is None or (isinstance(conf_id, str) and not conf_id.strip()):
                fallback_conf_id = (
                    conference.get('participationId')
                    or s_info.get('participationId')
                    or s_info.get('streamId')
                    or stream_data.get('id')
                    or data.get('id')
                    or data.get('participationId')
                    or f'url:{url}'
                )
                conf_id = f'fallback:{fallback_conf_id}'
            
            sk = (conf_id, is_screenshare)
            if stream_id:
                stream_id_to_sk[stream_id] = sk

            if sk not in streams:
                meta = conf_meta.get(sk[0], {})
                if not meta and conference.get('id') is not None:
                    meta = conf_meta.get(conference.get('id'), {})
                conf_user = conference.get('user', {}) if isinstance(conference.get('user'), dict) else {}
                user_id = meta.get('user_id') if meta.get('user_id') is not None else conf_user.get('id')
                user_name = meta.get('user_name') or conf_user.get('nickname', '')
                is_admin = (
                    user_id is not None
                    and admin_user_id is not None
                    and str(user_id) == str(admin_user_id)
                )
                streams[sk] = {'conference_id': sk[0], 'is_screenshare': is_screenshare, 'user_id': user_id, 'user_name': user_name, 'conf_has_video': meta.get('has_video', False), 'conf_has_audio': meta.get('has_audio', False), 'is_admin': is_admin, 'clips': []}
            
            # Truncate previous clip in this stream if it hasn't ended
            if streams[sk]['clips']:
                prev = streams[sk]['clips'][-1]
                if prev['duration'] == 0:
                    prev['duration'] = max(0, rel_time - prev['relative_time'])
            
            streams[sk]['clips'].append({'url': url, 'relative_time': rel_time, 'file_path': None, 'duration': 0, 'width': 0, 'height': 0, 'has_video': False, 'has_audio': False, 'stream_id': stream_id})

        elif module in ('conference.stream.delete', 'screensharing.stream.delete'):
            stream_id = data.get('id')
            if stream_id and stream_id in stream_id_to_sk:
                sk = stream_id_to_sk[stream_id]
                if streams[sk]['clips']:
                    prev = streams[sk]['clips'][-1]
                    if prev['duration'] == 0:
                        prev['duration'] = max(0, rel_time - prev['relative_time'])
        
        elif module in ('conference.update', 'screensharing.update') and data.get('isDeleted'):
            conf_id = data.get('id')
            is_screenshare = (module == 'screensharing.update')
            sk = (conf_id, is_screenshare)
            if sk in streams and streams[sk]['clips']:
                prev = streams[sk]['clips'][-1]
                if prev['duration'] == 0:
                    prev['duration'] = max(0, rel_time - prev['relative_time'])

    for s in streams.values(): s['clips'].sort(key=lambda c: c['relative_time'])
    return streams, admin_user_id

def parse_presentation_timeline(json_data: dict) -> Tuple[List[str], List[Tuple[float, float, str]]]:
    event_logs = json_data.get('eventLogs', [])
    pres_events = [e for e in event_logs if str(e.get('module', '')).startswith('presentation.')]
    slides, slide_timeline = [], []
    if not pres_events: return slides, slide_timeline

    deck_slides_by_key: Dict[str, List[str]] = {}
    active_deck_key = None

    def _deck_key_from_data(data: dict) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        file_reference = data.get('fileReference')
        if not isinstance(file_reference, dict):
            return None
        file_node = file_reference.get('file')
        if isinstance(file_node, dict):
            for key in ('id', 'uuid', 'hash', 'url'):
                value = file_node.get(key)
                if value is not None:
                    text = str(value).strip()
                    if text:
                        return text
        for key in ('id', 'uuid', 'hash'):
            value = file_reference.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
        return None

    def _slides_from_data(data: dict) -> List[str]:
        if not isinstance(data, dict):
            return []
        slide_list = data.get('fileReference', {}).get('file', {}).get('slides', [])
        if not isinstance(slide_list, list):
            return []
        
        res = []
        for item in slide_list:
            if isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, dict) and sub_item.get('url'):
                        res.append(sub_item.get('url'))
            elif isinstance(item, dict) and item.get('url'):
                res.append(item.get('url'))
        return res

    start, current_url = None, None
    for event in pres_events:
        d = event.get('data', {})
        t = event.get('relativeTime', 0)

        deck_key = _deck_key_from_data(d)
        new_slides = _slides_from_data(d)
        if new_slides and not deck_key:
            deck_key = 'slides:' + '|'.join(new_slides)
        if new_slides:
            if deck_key:
                deck_slides_by_key[deck_key] = new_slides
            if not slides:
                slides = new_slides
            else:
                slides = list(dict.fromkeys(slides + new_slides))

        if deck_key:
            active_deck_key = deck_key

        active_deck_slides = deck_slides_by_key.get(active_deck_key, []) if active_deck_key else []
        is_active = bool(d.get('isActive', True))
        new_url = _extract_slide_url(d, slides, current_url, deck_slides=active_deck_slides)
        if is_active:
            if start is None:
                start = t
                current_url = new_url
            elif new_url and new_url != current_url:
                if current_url:
                    slide_timeline.append((start, t, current_url))
                start = t
                current_url = new_url
        elif start is not None:
            if current_url:
                slide_timeline.append((start, t, current_url))
            start, current_url = None, None
    if start is not None and current_url:
        slide_timeline.append((start, float(json_data.get('duration', 0)), current_url))
    timeline_urls = [u for _, _, u in slide_timeline if u]
    if timeline_urls:
        slides = list(dict.fromkeys(timeline_urls + [u for u in slides if u]))
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


@contextmanager
def _tqdm_console_logging(root_logger: logging.Logger, formatter: logging.Formatter):
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(formatter)

    console_handlers = [
        h for h in list(root_logger.handlers)
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]

    for handler in console_handlers:
        root_logger.removeHandler(handler)
    root_logger.addHandler(tqdm_handler)

    try:
        yield
    finally:
        root_logger.removeHandler(tqdm_handler)
        for handler in console_handlers:
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)

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

def download_and_probe_all(streams: dict, directory: str, max_workers: int = None, progress_callback: Optional[Callable[[int, int], None]] = None):
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
                    if progress_callback:
                        progress_callback(pbar.n, len(all_clips))
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
                c_start = c['relative_time']
                c_end = c['relative_time'] + c['duration']
                if start_time <= c_start <= total_duration: change_points.add(c_start)
                if start_time <= c_end <= total_duration: change_points.add(c_end)
    for s_s, s_e, _ in (slide_timeline or []):
        if start_time <= s_s <= total_duration: change_points.add(s_s)
        if start_time <= s_e <= total_duration: change_points.add(s_e)
    for t in event_times:
        change_points.add(t)
    sorted_p = sorted(list(change_points))
    filtered = [sorted_p[0]]
    for p in sorted_p[1:]:
        if p - filtered[-1] >= MIN_SEGMENT_DURATION or p == total_duration: filtered.append(p)
    segments = []
    for i in range(len(filtered)-1):
        t_s, t_e = filtered[i], filtered[i+1]
        t_probe = min(t_e - 0.001, t_s + min(0.2, max(0.01, (t_e - t_s) / 2)))
        t_m = (t_s + t_e) / 2
        active = []
        for sk, s in streams.items():
            clip, seek = _find_clip_at_time(s, t_probe)
            if not clip:
                clip, seek = _find_clip_at_time(s, t_m)
            if clip: active.append((sk, clip, seek))
        screenshare = _choose_primary_screenshare(active, streams, admin_user_id)
        all_cameras = [a for a in active if not streams[a[0]]['is_screenshare']]
        audio_sources = [a for a in active if a[1]['has_audio']]
        lecturer = next((c for c in all_cameras if _is_lecturer_stream(streams[c[0]], admin_user_id)), None)
        cameras = list(all_cameras)
        preserve_speaking_order = bool(hide_silent and speech_timelines)
        if hide_silent and speech_timelines:
            speaking_overlap = {c[0]: _speech_overlap_duration(speech_timelines.get(c[0], []), t_s, t_e) for c in all_cameras}
            speaking = [c for c in all_cameras if speaking_overlap.get(c[0], 0.0) > 0.05]
            speaking.sort(key=lambda c: (-speaking_overlap.get(c[0], 0.0), streams[c[0]].get('user_name', '').lower()))
            cameras = ([lecturer] if lecturer else []) + [c for c in speaking if not lecturer or c[0] != lecturer[0]]
        cameras = [c for c in cameras if c[1]['has_video'] or c[1]['has_audio']]
        if lecturer and not any(c[0] == lecturer[0] for c in cameras):
            if lecturer[1]['has_video'] or lecturer[1]['has_audio']:
                cameras = [lecturer] + cameras
        if not preserve_speaking_order:
            cameras.sort(key=lambda c: (0 if _is_lecturer_stream(streams[c[0]], admin_user_id) else 1, streams[c[0]].get('user_name', '').lower()))
        slide = next((path for s_s, s_e, path in (slide_timeline or []) if s_s <= t_probe < s_e), None)
        screenshare_has_visual = bool(screenshare and screenshare[1].get('has_video'))
        if screenshare and slide:
            if screenshare_has_visual:
                slide = None
            else:
                screenshare = None
        elif screenshare and not screenshare_has_visual:
            screenshare = None
        chat_version = bisect_right(event_times, t_probe)
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
    main_w = (int(out_w * 0.75) // 2) * 2
    sidebar_w = out_w - main_w
    gap = max(8, out_h // 90)
    cam_w = (sidebar_w - 3 * gap) // 2
    cam_w = max(100, (cam_w // 2) * 2)
    cam_h = max(90, int(cam_w * 0.75))
    cam_h = (cam_h // 2) * 2
    chat_y = 2 * cam_h + 3 * gap
    chat_h = max(120, out_h - chat_y - gap)
    chat_w = max(120, ((2 * cam_w + gap - max(4, gap // 2)) // 2) * 2)
    chat_x = main_w + (sidebar_w - chat_w) // 2
    cam_slots = [
        (main_w + gap, gap),
        (main_w + 2 * gap + cam_w, gap),
        (main_w + gap, 2 * gap + cam_h),
        (main_w + 2 * gap + cam_w, 2 * gap + cam_h),
    ]
    cam_visual_count = 0
    max_cam_slots = 4
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
            img_path = _create_name_placeholder(user_name, os.path.dirname(output_path), cam_w, cam_h)
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
            filter_parts.append(f'[{idx}:v]{pts}scale={main_w}:{out_h}:force_original_aspect_ratio=decrease,pad={main_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,scale={main_w}:{out_h}[m_v]')
            filter_parts.append(f'[{curr_v}][m_v]overlay=0:0:eof_action=repeat[v{ov_idx}]')
        elif type == 'main_no_pts':
            filter_parts.append(f'[{idx}:v]scale={cam_w}:{cam_h},setsar=1[c{idx}]')
            slot_x, slot_y = cam_slots[min(cam_count, len(cam_slots)-1)]
            filter_parts.append(f'[{curr_v}][c{idx}]overlay={slot_x}:{slot_y}:eof_action=repeat[v{ov_idx}]')
            cam_count += 1
        else:
            filter_parts.append(f'[{idx}:v]scale={cam_w}:{cam_h}:force_original_aspect_ratio=decrease,pad={cam_w}:{cam_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,scale={cam_w}:{cam_h}[c{idx}]')
            slot_x, slot_y = cam_slots[min(cam_count, len(cam_slots)-1)]
            filter_parts.append(f'[{curr_v}][c{idx}]overlay={slot_x}:{slot_y}:eof_action=repeat[v{ov_idx}]')
            cam_count += 1
        curr_v, ov_idx = f'v{ov_idx}', ov_idx + 1

    chat_wrap_chars = max(20, int((chat_w - 22) / 7.3))
    chat_max_lines = max(10, int((chat_h - 44) / 18))
    # Use probe time (middle of segment) for chat to match timeline split logic
    t_probe = t_start + duration / 2
    chat_txt = _build_chat_overlay_text(events, t_probe, max_lines=chat_max_lines, wrap_chars=chat_wrap_chars)
    chat_path = os.path.join(tmpdir, f'chat_{seg_index:05d}.txt')
    with open(chat_path, 'w', encoding='utf-8') as f:
        f.write(chat_txt)
    esc_chat_path = _escape_drawtext_path(chat_path)
    filter_parts.append(
        f"color=black@0.68:s={chat_w}x{chat_h}:d={duration:.3f},"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.10:t=2,"
        f"drawtext=text='CHAT / Q&A':fontcolor=white:fontsize=15:x=10:y=8,"
        f"drawtext=fontcolor=cyan:fontsize=13:line_spacing=5:x=10:y=32:textfile='{esc_chat_path}':reload=0[chat]"
    )
    filter_parts.append(f'[{curr_v}][chat]overlay={chat_x}:{chat_y}[v_fin]')
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

def render_all_segments(timeline: list, streams: dict, tmpdir: str, out_w: int = 1920, out_h: int = 1080, events: list = None, process_started_ts: Optional[float] = None, progress_callback: Optional[Callable[[int, int], None]] = None) -> list:
    global STOP_REQUESTED
    encoder, preset, max_workers = _detect_best_encoder(); cpu_count = os.cpu_count() or 4
    threads_per_worker = max(1, min(4, cpu_count // max_workers)); segment_files = [None] * len(timeline)
    render_started_ts = time.perf_counter()
    render_started_at = datetime.datetime.now()
    logging.info(f'Rendering started at {render_started_at.strftime("%Y-%m-%d %H:%M:%S")} ({len(timeline)} segments) | encoder={encoder}, preset={preset}, workers={max_workers}, threads/worker={threads_per_worker}')
    root_logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
    with _tqdm_console_logging(root_logger, formatter):
        tqdm_args = {"total": len(timeline), "desc": "Rendering", "unit": "seg", "ascii": True, "mininterval": 0.5, "leave": True}
        with tqdm.tqdm(**tqdm_args) as pbar:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            completed = 0
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
                        interrupted = 'interrupted' in str(e).lower()
                        if interrupted:
                            STOP_REQUESTED = True
                        elif not STOP_REQUESTED:
                            logging.error(f'Critical: Segment {idx} failed: {e}')
                        executor.shutdown(wait=False, cancel_futures=True); raise
                    pbar.update(1)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(timeline))
                    if completed % 10 == 0 or completed == len(timeline):
                        render_elapsed = _format_elapsed(time.perf_counter() - render_started_ts)
                        if process_started_ts is not None:
                            total_elapsed = _format_elapsed(time.perf_counter() - process_started_ts)
                            logging.info(f'Rendering progress: {completed}/{len(timeline)} segments | render elapsed {render_elapsed} | total elapsed {total_elapsed}')
                        else:
                            logging.info(f'Rendering progress: {completed}/{len(timeline)} segments | render elapsed {render_elapsed}')
            finally: executor.shutdown(wait=True)
    if STOP_REQUESTED: raise RuntimeError("interrupted")
    render_finished_at = datetime.datetime.now()
    logging.info(f'Rendering finished at {render_finished_at.strftime("%Y-%m-%d %H:%M:%S")} | render elapsed {_format_elapsed(time.perf_counter() - render_started_ts)}')
    return [f for f in segment_files if f]

def concat_segments(segment_files: list, output_path: str):
    if STOP_REQUESTED: return
    list_path = os.path.join(os.path.dirname(segment_files[0]), 'concat.txt')
    with open(list_path, 'w') as f:
        for s in segment_files: f.write(f"file '{s.replace(chr(92), '/')}'\n")
    _run_ffmpeg(['-y', '-f', 'concat', '-safe', '0', '-fflags', '+genpts', '-i', list_path, '-c', 'copy', '-avoid_negative_ts', 'make_zero', '-movflags', '+faststart', output_path], desc='concat')


def _resolve_output_resolution(quality: str) -> Tuple[int, int]:
    return {"1080p": (1920, 1080), "720p": (1280, 720)}.get(quality, (1920, 1080))


def _resolve_total_duration(json_data: dict, max_duration, start_time: float) -> float:
    total_duration = float(json_data.get('duration', 0))
    if max_duration:
        total_duration = min(total_duration, start_time + max_duration)
    if total_duration <= start_time:
        raise ValueError(f'Invalid time window: start_time={start_time}, end_time={total_duration}')
    return total_duration


def _prepare_stream_clip_flags(streams: dict, hide_silent: bool):
    for stream in streams.values():
        for clip in stream['clips']:
            if stream['is_screenshare']:
                clip['_is_screenshare'] = True
            elif hide_silent:
                clip['_run_vad'] = True


def _collect_slide_urls(slides: List[str], slide_timeline: Optional[list]) -> List[str]:
    timeline_urls = [url for _, _, url in (slide_timeline or []) if url]
    return list(dict.fromkeys([u for u in slides if u] + timeline_urls))


def _download_slide_images(directory: str, slide_urls: List[str]) -> Dict[str, str]:
    slide_map = {}
    if not slide_urls:
        return slide_map

    with create_shared_client() as client:
        for i, url in enumerate(slide_urls):
            if STOP_REQUESTED:
                break
            path = os.path.join(directory, f'slide_{i:03d}.jpg')
            if not os.path.exists(path):
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    with open(path, 'wb') as file_handle:
                        file_handle.write(response.content)
                except Exception:
                    continue
            slide_map[url] = path
    return slide_map


def _materialize_slide_timeline(directory: str, slides: List[str], slide_timeline: Optional[list]) -> list:
    slide_urls = _collect_slide_urls(slides, slide_timeline)
    slide_map = _download_slide_images(directory, slide_urls)
    return [(s, e, slide_map[url]) for s, e, url in (slide_timeline or []) if url in slide_map]


def _timeline_coverage(intervals: Optional[list], start_time: float, end_time: float) -> float:
    if not intervals or end_time <= start_time:
        return 0.0
    total = 0.0
    for start, end in intervals:
        overlap = min(float(end), end_time) - max(float(start), start_time)
        if overlap > 0:
            total += overlap
    return total


def _log_source_switch_diagnostics(
    json_data: dict,
    streams: dict,
    raw_slide_timeline: Optional[list],
    start_time: float,
    total_duration: float,
):
    pres_events = [e for e in json_data.get('eventLogs', []) if str(e.get('module', '')).startswith('presentation.')]
    slide_intervals = [(s, e) for s, e, _ in (raw_slide_timeline or [])]
    slide_coverage = _timeline_coverage(slide_intervals, start_time, total_duration)

    screenshare_intervals = []
    for stream in streams.values():
        if not stream.get('is_screenshare'):
            continue
        for clip in stream.get('clips', []):
            clip_start = float(clip.get('relative_time', 0.0))
            clip_end = clip_start + max(0.0, float(clip.get('duration', 0.0)))
            if clip_end > clip_start:
                screenshare_intervals.append((clip_start, clip_end))

    screenshare_coverage = _timeline_coverage(screenshare_intervals, start_time, total_duration)
    overlap = 0.0
    for s_start, s_end in screenshare_intervals:
        for p_start, p_end in slide_intervals:
            overlap += max(0.0, min(s_end, p_end, total_duration) - max(s_start, p_start, start_time))

    file_reference_changes = 0
    known_file_refs = set()
    for event in pres_events:
        data = event.get('data', {})
        file_reference = data.get('fileReference', {}) if isinstance(data, dict) else {}
        file_id = file_reference.get('file', {}).get('id') or file_reference.get('id')
        if file_id:
            known_file_refs.add(str(file_id))
            file_reference_changes = max(file_reference_changes, len(known_file_refs))

    logging.info(
        'Source diagnostics: screenshare_coverage=%.1fs, slide_coverage=%.1fs, overlap=%.1fs, presentation_events=%d, presentation_files=%d',
        screenshare_coverage,
        slide_coverage,
        overlap,
        len(pres_events),
        file_reference_changes,
    )

    if pres_events and not raw_slide_timeline:
        logging.warning('Presentation events exist but no slide timeline could be built; slide sync can be inaccurate.')
    if overlap > 30.0:
        logging.warning(
            'Detected %.1fs where presentation and screenshare are both active. Presenter may switch visible source; timing can drift without explicit stage-switch events.',
            overlap,
        )
    if file_reference_changes >= 2:
        logging.warning(
            'Detected %d different presentation file references. Source switches between slide decks are possible; verify timeline around switch points.',
            file_reference_changes,
        )


def process_composite_video(directory: str, json_data: dict, output_path: str, max_duration=None, hide_silent: bool = False, start_time: float = 0, quality: str = "1080p", status_file: Optional[str] = None):
    global STOP_REQUESTED
    STOP_REQUESTED = False 
    process_started_ts = time.perf_counter()
    process_started_at = datetime.datetime.now()
    status_payload = {
        'state': 'running',
        'stage': 'initializing',
        'started_at': process_started_at.isoformat(timespec='seconds'),
        'updated_at': process_started_at.isoformat(timespec='seconds'),
        'output_path': os.path.abspath(output_path),
        'start_time_sec': float(start_time),
    }

    def _update_status(state: Optional[str] = None, stage: Optional[str] = None, progress: Optional[dict] = None, message: Optional[str] = None):
        if state:
            status_payload['state'] = state
        if stage:
            status_payload['stage'] = stage
        if progress is not None:
            status_payload['progress'] = progress
        if message is not None:
            status_payload['message'] = message
        status_payload['updated_at'] = datetime.datetime.now().isoformat(timespec='seconds')
        status_payload['elapsed'] = _format_elapsed(time.perf_counter() - process_started_ts)
        _write_status_file(status_file, status_payload)

    _update_status(stage='initializing')
    logging.info(f'Process started at {process_started_at.strftime("%Y-%m-%d %H:%M:%S")}')
    try:
        out_w, out_h = _resolve_output_resolution(quality)
        total_duration = _resolve_total_duration(json_data, max_duration, start_time)
        _update_status(stage='preparing', progress={'current': 0, 'total': 1, 'unit': 'step'})
        logging.info(f'Target timeline: start={start_time:.1f}s, end={total_duration:.1f}s, duration={max(0.0, total_duration-start_time):.1f}s')
        chat, questions = parse_chat_and_questions(json_data)
        all_events = sorted(chat + questions, key=lambda x: x['time'])
        export_chat_files(chat, questions, directory, os.path.basename(output_path).replace('.mp4', ''))
        streams, admin_id = parse_event_logs(json_data)
        if not streams:
            _update_status(state='completed', stage='done', message='No media streams found.')
            return
        _prepare_stream_clip_flags(streams, hide_silent)

        _update_status(stage='downloading', progress={'current': 0, 'total': sum(len(s['clips']) for s in streams.values()), 'unit': 'file'})

        def _on_download_progress(current: int, total: int):
            if current == total or current % 20 == 0:
                _update_status(stage='downloading', progress={'current': current, 'total': total, 'unit': 'file'})

        download_and_probe_all(streams, directory, progress_callback=_on_download_progress)
        logging.info(f'Files processed | elapsed {_format_elapsed(time.perf_counter() - process_started_ts)}')
        _reclassify_screenshare_by_dimensions(streams)
        slides, s_timeline = parse_presentation_timeline(json_data)
        _log_source_switch_diagnostics(json_data, streams, s_timeline, start_time, total_duration)
        if slides:
            _update_status(stage='slides', progress={'current': 0, 'total': len(slides), 'unit': 'slide'})
            s_timeline = _materialize_slide_timeline(directory, slides, s_timeline)
            _update_status(stage='slides', progress={'current': len(slides), 'total': len(slides), 'unit': 'slide'})
        missing_slide_images = sum(1 for _, _, path in s_timeline if not os.path.exists(path)) if s_timeline else 0
        if missing_slide_images:
            logging.warning('Slide timeline contains %d missing local slide images; these intervals may render as black.', missing_slide_images)
        if STOP_REQUESTED:
            raise RuntimeError("interrupted")
        speech_timelines = {sk: _build_stream_speech_timeline(s) for sk, s in streams.items() if not s['is_screenshare']} if hide_silent else {}
        timeline = compute_layout_timeline(streams, total_duration, admin_id, hide_silent, speech_timelines, start_time, s_timeline, all_events)
        if not timeline:
            _update_status(state='completed', stage='done', message='No timeline segments to render.')
            return
        _update_status(stage='timeline', progress={'current': len(timeline), 'total': len(timeline), 'unit': 'segment'})
        logging.info(f'Timeline ready: {len(timeline)} segments | elapsed {_format_elapsed(time.perf_counter() - process_started_ts)}')
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            for i, segment in enumerate(timeline[:20]):
                main_source = 'screen' if segment.get('screenshare') else ('slide' if segment.get('slide_image') else 'none')
                logging.debug(
                    'seg[%d] %.2f-%.2f source=%s cameras=%d audio=%d chat=%d',
                    i,
                    segment['start'],
                    segment['end'],
                    main_source,
                    len(segment.get('cameras', [])),
                    len(segment.get('audio_sources', [])),
                    segment.get('chat_version', 0),
                )
            if len(timeline) > 20:
                logging.debug('... %d more segments omitted from debug output', len(timeline) - 20)
        if hide_silent:
            active = {streams[sk]['user_name'] for sk, t in speech_timelines.items() if t}
            logging.info(f'Active participants (with speech): {len(active)}')

        _update_status(stage='rendering', progress={'current': 0, 'total': len(timeline), 'unit': 'segment'})

        def _on_render_progress(current: int, total: int):
            if current == total or current % 10 == 0:
                _update_status(stage='rendering', progress={'current': current, 'total': total, 'unit': 'segment'})

        with tempfile.TemporaryDirectory() as tmp:
            files = render_all_segments(
                timeline,
                streams,
                tmp,
                out_w,
                out_h,
                events=all_events,
                process_started_ts=process_started_ts,
                progress_callback=_on_render_progress,
            )
            if files:
                _update_status(stage='concatenating')
                concat_segments(files, output_path)
        logging.info(f'Done: {output_path}')
        process_finished_at = datetime.datetime.now()
        _update_status(state='completed', stage='done', progress={'current': 1, 'total': 1, 'unit': 'job'})
        logging.info(f'Process finished at {process_finished_at.strftime("%Y-%m-%d %H:%M:%S")} | total elapsed {_format_elapsed(time.perf_counter() - process_started_ts)}')
    except Exception as exc:
        final_state = 'interrupted' if 'interrupted' in str(exc).lower() else 'failed'
        _update_status(state=final_state, stage='error', message=str(exc))
        raise

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
