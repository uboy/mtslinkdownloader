import time

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
