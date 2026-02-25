# mtslinkdownloader Functional Specification

## 1. Purpose

`mtslinkdownloader` is a CLI and Python package that downloads MTS Link webinar recordings, reconstructs a composite timeline from event logs, and renders a final MP4 with adaptive layout.

## 2. Supported Input

- Public record URL:
  - `https://<host>.mts-link.ru/.../record-new/{event_session_id}/record-file/{record_id}`
- Quick meeting URL (without `record-file`):
  - `https://<host>.mts-link.ru/.../record-new/{event_session_id}`
- Optional private access cookie:
  - `sessionId` passed via `--session-id`

## 3. Public Interfaces

- CLI command:
  - `mtslinkdownloader <url> [--session-id ...] [--hide-silent] [--max-duration ...] [--start-time ...]`
- Python API:
  - `fetch_webinar_data(event_sessions, record_id=None, session_id=None, max_duration=None, hide_silent=False, start_time=0)`

## 4. Processing Pipeline

1. Parse webinar URL into `event_session_id` and optional `record_id`.
2. Build metadata endpoint URL:
   - quick meeting: `/api/eventsessions/{event_session_id}/record`
   - regular record: `/api/event-sessions/{event_session_id}/record-files/{record_id}/flow`
3. Load JSON metadata (`eventLogs`, `duration`, record name).
4. Parse streams from event logs:
   - detect camera vs screenshare streams
   - detect admin user id (`eventsession.start`) as a weak tie-break signal
   - build stream key as `(conference_id, is_screenshare)` with fallback id (`fallback:<...>`) when `conference_id` is missing
5. Download media chunks concurrently.
6. Probe each chunk with `ffprobe`:
   - duration, video/audio presence, dimensions
   - optional black-frame rejection for camera streams
   - optional VAD intervals for `--hide-silent`
7. Reclassify high-resolution streams as screenshare when needed.
8. Parse presentation slides timeline (`presentation.update`) and download slide images.
9. Build layout timeline segments from stream boundaries and slide boundaries.
10. Render each segment with `ffmpeg`:
    - Mode A: screenshare main area + camera sidebar
    - Mode B: camera grid layout
    - mix audio from active tracks
11. Concatenate segment files into final MP4.

### 4.1 Primary Visual Source Selection

Per segment, the main visual source is selected as follows:
- if active screenshare clip has video, render screenshare on the main canvas;
- otherwise, use slide image from `presentation.*` timeline (if available);
- if neither is available, render black background on main canvas.

When multiple screenshares are active, selection is deterministic and uses media signals first
(`has_video`, frame area, audio, clip start time). Lecturer/admin affinity is only a tie-break.

## 5. Output

- Working directory: sanitized webinar name.
- Final file: `<sanitized_name>/<sanitized_name>.mp4`.
- Log file: `logs/mtslinkdownloader.log`.

## 6. Non-Functional Characteristics

- Parallel downloads and segment rendering with thread pools.
- Uses `httpx` connection pooling.
- Uses system `ffmpeg`/`ffprobe` if available, otherwise bundled binary from `imageio_ffmpeg`.
- Fail-fast behavior on invalid URL or failed metadata fetch.

## 7. Known Constraints

- Requires ffmpeg-compatible environment for rendering.
- Input URL must be from `mts-link.ru` domain format.
- Presentation and screenshare overlap is inferred heuristically from event logs; explicit stage-switch events may be absent.
- ADMIN role metadata can be stale/incomplete; it is treated as heuristic, not source-of-truth for visible source.

## 8. Refactoring and Rename Scope

- Project/package rename:
  - `mtslinker` -> `mtslinkdownloader`
- Packaging updates:
  - distribution name and console entrypoint
- Runtime updates:
  - imports and logger filename
- Container/docs updates:
  - Docker entrypoint, compose service/container name, README examples
- Reliability refactor:
  - strict URL validation in CLI before processing
