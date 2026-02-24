import logging
import io
import re
import subprocess
import time

import pytest

from mtslinkdownloader import processor


def _make_clip(path: str, *, has_video=True, has_audio=True, duration=10.0, proxy_path=None):
    clip = {
        "relative_time": 0.0,
        "duration": duration,
        "has_video": has_video,
        "has_audio": has_audio,
        "file_path": path,
    }
    if proxy_path:
        clip["proxy_path"] = proxy_path
    return clip


def _has_ffmpeg() -> bool:
    ffmpeg = processor._get_ffmpeg()
    if not ffmpeg:
        return False
    try:
        result = subprocess.run([ffmpeg, "-version"], capture_output=True)
        return result.returncode == 0
    except Exception:
        return False


def _sample_rgb(video_path: str, x: int, y: int, ts: float = 0.5):
    ffmpeg = processor._get_ffmpeg()
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{ts:.3f}",
        "-i",
        video_path,
        "-vf",
        f"crop=1:1:{x}:{y},format=rgb24",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    if len(result.stdout) < 3:
        raise AssertionError("Unable to sample pixel from rendered frame")
    return result.stdout[0], result.stdout[1], result.stdout[2]


def _create_color_tone_clip(ffmpeg: str, output_path, color: str, frequency: int, duration: float = 1.2):
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=640x360:d={duration}:r=30",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={duration}:sample_rate=44100",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def _create_color_image(ffmpeg: str, output_path, color: str):
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=1280x720:d=1",
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def test_render_segment_falls_back_to_black_when_no_inputs(tmp_path, monkeypatch):
    captured = {}

    def fake_run_ffmpeg(args, desc="", timeout=None):
        captured["args"] = args
        captured["desc"] = desc
        captured["timeout"] = timeout

    monkeypatch.setattr(processor, "_run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(processor, "STOP_REQUESTED", False)

    segment = {
        "start": 0.0,
        "end": 1.0,
        "screenshare": None,
        "cameras": [],
        "audio_sources": [],
        "slide_image": None,
        "chat_version": 0,
    }

    output = processor._render_segment(
        seg_index=0,
        segment=segment,
        streams={},
        tmpdir=str(tmp_path),
        threads=2,
    )

    assert output.endswith("seg_00000.mp4")
    assert captured["desc"] == "black 0"
    assert any("color=black" in arg for arg in captured["args"])


def test_render_segment_builds_chat_overlay_and_audio_mix(tmp_path, monkeypatch):
    captured = {}

    def fake_run_ffmpeg(args, desc="", timeout=None):
        captured["args"] = args
        captured["desc"] = desc
        captured["timeout"] = timeout

    monkeypatch.setattr(processor, "_run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(processor, "_build_chat_overlay_text", lambda *_, **__: "line1\nline2")
    monkeypatch.setattr(processor, "STOP_REQUESTED", False)

    stream_key = ("cam1", False)
    clip = _make_clip("cam.mp4", proxy_path="cam_proxy.mp4")
    streams = {
        stream_key: {
            "user_name": "User",
            "is_screenshare": False,
            "clips": [clip],
        }
    }

    segment = {
        "start": 0.0,
        "end": 2.0,
        "screenshare": None,
        "cameras": [(stream_key, clip, 0.0)],
        "audio_sources": [(stream_key, clip, 0.0)],
        "slide_image": "slide.jpg",
        "chat_version": 1,
    }

    processor._render_segment(
        seg_index=1,
        segment=segment,
        streams=streams,
        tmpdir=str(tmp_path),
        threads=1,
        out_w=1280,
        out_h=720,
        events=[{"time": 0.0, "type": "CHAT", "user": "u", "text": "t"}],
    )

    chat_file = tmp_path / "chat_00001.txt"
    assert chat_file.exists()
    assert chat_file.read_text(encoding="utf-8") == "line1\nline2"
    assert captured["desc"] == "seg 1"

    filter_idx = captured["args"].index("-filter_complex")
    filter_expr = captured["args"][filter_idx + 1]
    assert "drawtext=text='CHAT / Q&A'" in filter_expr
    assert "amix=inputs=1" in filter_expr


def test_render_all_segments_returns_results_in_timeline_order(tmp_path, monkeypatch):
    monkeypatch.setattr(processor, "STOP_REQUESTED", False)
    monkeypatch.setattr(processor, "_detect_best_encoder", lambda: ("libx264", "ultrafast", 2))

    class _DummyTqdm:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, _n):
            return None

    monkeypatch.setattr(processor.tqdm, "tqdm", _DummyTqdm)

    def fake_render_segment(seg_index, *_args, **_kwargs):
        time.sleep(0.03 if seg_index == 0 else 0.0)
        return str(tmp_path / f"seg_{seg_index:05d}.mp4")

    monkeypatch.setattr(processor, "_render_segment", fake_render_segment)

    timeline = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.0, "end": 2.0},
        {"start": 2.0, "end": 3.0},
    ]
    result = processor.render_all_segments(timeline, {}, str(tmp_path))

    assert result == [
        str(tmp_path / "seg_00000.mp4"),
        str(tmp_path / "seg_00001.mp4"),
        str(tmp_path / "seg_00002.mp4"),
    ]


def test_render_all_segments_interrupted_suppresses_critical_error_log(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(processor, "STOP_REQUESTED", False)
    monkeypatch.setattr(processor, "_detect_best_encoder", lambda: ("libx264", "ultrafast", 1))

    class _DummyTqdm:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, _n):
            return None

    monkeypatch.setattr(processor.tqdm, "tqdm", _DummyTqdm)
    monkeypatch.setattr(
        processor,
        "_render_segment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )

    timeline = [{"start": 0.0, "end": 1.0}]
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="interrupted"):
            processor.render_all_segments(timeline, {}, str(tmp_path))

    assert processor.STOP_REQUESTED is True
    assert not any("Critical: Segment" in m for m in caplog.messages)


def test_tqdm_console_logging_temporarily_replaces_console_handlers(tmp_path):
    logger = logging.getLogger("mtslinkdownloader.test.tqdm_context")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.handlers = []

    stream_handler = logging.StreamHandler(io.StringIO())
    file_handler = logging.FileHandler(tmp_path / "ctx.log")
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    formatter = logging.Formatter("%(message)s")

    with processor._tqdm_console_logging(logger, formatter):
        assert stream_handler not in logger.handlers
        assert file_handler in logger.handlers
        assert any(isinstance(h, processor.TqdmLoggingHandler) for h in logger.handlers)

    assert stream_handler in logger.handlers
    assert file_handler in logger.handlers
    assert not any(isinstance(h, processor.TqdmLoggingHandler) for h in logger.handlers)

    file_handler.close()


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg is not available")
def test_render_segment_end_to_end_layout_and_audio(tmp_path, monkeypatch):
    ffmpeg = processor._get_ffmpeg()
    monkeypatch.setattr(processor, "STOP_REQUESTED", False)

    lecturer_video = tmp_path / "lecturer.mp4"
    participant_video = tmp_path / "participant.mp4"
    slide_image = tmp_path / "slide.jpg"

    _create_color_tone_clip(ffmpeg, lecturer_video, "red", 440, duration=1.2)
    _create_color_tone_clip(ffmpeg, participant_video, "green", 880, duration=1.2)
    _create_color_image(ffmpeg, slide_image, "blue")

    lecturer_key = ("lecturer", False)
    participant_key = ("participant", False)
    streams = {
        lecturer_key: {
            "is_admin": True,
            "is_screenshare": False,
            "user_id": 1,
            "user_name": "Lecturer",
            "clips": [ _make_clip(str(lecturer_video), duration=1.2) ],
        },
        participant_key: {
            "is_admin": False,
            "is_screenshare": False,
            "user_id": 2,
            "user_name": "Participant",
            "clips": [ _make_clip(str(participant_video), duration=1.2) ],
        },
    }
    events = [{"time": 0.0, "type": "CHAT", "user": "Alice", "text": "hello"}]

    timeline = processor.compute_layout_timeline(
        streams=streams,
        total_duration=1.0,
        admin_user_id=1,
        start_time=0.0,
        slide_timeline=[(0.0, 1.0, str(slide_image))],
        events=events,
    )

    assert len(timeline) == 1
    assert timeline[0]["cameras"][0][0] == lecturer_key

    output = processor._render_segment(
        seg_index=3,
        segment=timeline[0],
        streams=streams,
        tmpdir=str(tmp_path),
        threads=1,
        encoder="libx264",
        preset="ultrafast",
        out_w=1280,
        out_h=720,
        events=events,
    )

    info = processor._probe_media(output)
    assert info["has_video"] is True
    assert info["has_audio"] is True
    assert info["width"] == 1280
    assert info["height"] == 720

    # Main area: blue slide.
    r, g, b = _sample_rgb(output, 200, 200)
    assert b > r + 25 and b > g + 25

    # Sidebar top-left camera slot: lecturer (red).
    r, g, b = _sample_rgb(output, 1042, 63)
    assert r > g + 25 and r > b + 25

    # Sidebar top-right camera slot: participant (green).
    r, g, b = _sample_rgb(output, 1198, 63)
    assert g > r + 25 and g > b + 25

    # Chat block border should brighten pixel over black background.
    r, g, b = _sample_rgb(output, 1046, 245)
    assert (r + g + b) > 15

    volume_probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", output, "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", volume_probe.stderr)
    assert match is not None
    assert float(match.group(1)) > -45.0


def test_render_segment_keeps_audio_sources_even_for_hidden_camera(tmp_path, monkeypatch):
    captured = {}

    def fake_run_ffmpeg(args, desc="", timeout=None):
        captured["args"] = args
        captured["desc"] = desc
        captured["timeout"] = timeout

    monkeypatch.setattr(processor, "_run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(processor, "_build_chat_overlay_text", lambda *_, **__: "chat")
    monkeypatch.setattr(processor, "STOP_REQUESTED", False)

    lecturer_key = ("lecturer", False)
    hidden_key = ("hidden", False)
    lecturer_clip = _make_clip("lecturer.mp4")
    hidden_clip = _make_clip("hidden.mp4")
    streams = {
        lecturer_key: {"user_name": "Lecturer", "is_screenshare": False, "clips": [lecturer_clip]},
        hidden_key: {"user_name": "Hidden", "is_screenshare": False, "clips": [hidden_clip]},
    }

    segment = {
        "start": 0.0,
        "end": 1.0,
        "screenshare": None,
        "cameras": [(lecturer_key, lecturer_clip, 0.0)],
        "audio_sources": [(lecturer_key, lecturer_clip, 0.0), (hidden_key, hidden_clip, 0.0)],
        "slide_image": "slide.jpg",
        "chat_version": 0,
    }

    processor._render_segment(
        seg_index=4,
        segment=segment,
        streams=streams,
        tmpdir=str(tmp_path),
        threads=1,
        out_w=1280,
        out_h=720,
        events=[],
    )

    filter_idx = captured["args"].index("-filter_complex")
    filter_expr = captured["args"][filter_idx + 1]
    assert "amix=inputs=2" in filter_expr


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg is not available")
def test_acceptance_window_slides_chat_and_concat(tmp_path, monkeypatch):
    ffmpeg = processor._get_ffmpeg()
    monkeypatch.setattr(processor, "STOP_REQUESTED", False)
    monkeypatch.setattr(processor, "_detect_best_encoder", lambda: ("libx264", "ultrafast", 1))

    lecturer_video = tmp_path / "lecturer_long.mp4"
    participant_video = tmp_path / "participant_long.mp4"
    slide1 = tmp_path / "slide1.jpg"
    slide2 = tmp_path / "slide2.jpg"
    _create_color_tone_clip(ffmpeg, lecturer_video, "red", 440, duration=2.2)
    _create_color_tone_clip(ffmpeg, participant_video, "green", 880, duration=2.2)
    _create_color_image(ffmpeg, slide1, "blue")
    _create_color_image(ffmpeg, slide2, "yellow")

    lecturer_key = ("lecturer", False)
    participant_key = ("participant", False)
    streams = {
        lecturer_key: {
            "is_admin": True,
            "is_screenshare": False,
            "user_id": 1,
            "user_name": "Lecturer",
            "clips": [_make_clip(str(lecturer_video), duration=2.2)],
        },
        participant_key: {
            "is_admin": False,
            "is_screenshare": False,
            "user_id": 2,
            "user_name": "Participant",
            "clips": [_make_clip(str(participant_video), duration=2.2)],
        },
    }
    events = [{"time": 1.0, "type": "CHAT", "user": "Alice", "text": "state update"}]
    speech_timelines = {
        lecturer_key: [],
        participant_key: [],
    }

    timeline = processor.compute_layout_timeline(
        streams=streams,
        total_duration=1.7,
        admin_user_id=1,
        hide_silent=True,
        speech_timelines=speech_timelines,
        start_time=0.5,
        slide_timeline=[(0.5, 1.0, str(slide1)), (1.0, 1.7, str(slide2))],
        events=events,
    )

    assert len(timeline) == 2
    assert timeline[0]["start"] == 0.5
    assert timeline[-1]["end"] == 1.7
    assert [segment["chat_version"] for segment in timeline] == [0, 1]
    for segment in timeline:
        assert [c[0] for c in segment["cameras"]] == [lecturer_key]
        assert {a[0] for a in segment["audio_sources"]} == {lecturer_key, participant_key}

    segment_files = processor.render_all_segments(
        timeline=timeline,
        streams=streams,
        tmpdir=str(tmp_path),
        out_w=1280,
        out_h=720,
        events=events,
    )
    assert len(segment_files) == 2

    output_path = tmp_path / "acceptance_final.mp4"
    processor.concat_segments(segment_files, str(output_path))

    info = processor._probe_media(str(output_path))
    assert info["has_video"] is True
    assert info["has_audio"] is True
    assert 1.10 <= info["duration"] <= 1.35

    r, g, b = _sample_rgb(str(output_path), 120, 120, ts=0.2)
    assert b > r + 25 and b > g + 25
    r, g, b = _sample_rgb(str(output_path), 120, 120, ts=0.95)
    assert r > 100 and g > 100 and b < 120

    volume_probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(output_path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", volume_probe.stderr)
    assert match is not None
    assert float(match.group(1)) > -45.0
