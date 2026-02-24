from mtslinkdownloader.processor import (
    compute_layout_timeline,
    _build_chat_overlay_text,
    _format_elapsed,
    parse_chat_and_questions,
)


def _make_stream(is_admin=False, is_screenshare=False, user_id=None, user_name=""):
    return {
        "is_admin": is_admin,
        "is_screenshare": is_screenshare,
        "user_id": user_id,
        "user_name": user_name,
        "clips": [],
    }


def _make_clip(start=0.0, duration=10.0, has_video=True, has_audio=True, file_path="clip.mp4"):
    return {
        "relative_time": start,
        "duration": duration,
        "has_video": has_video,
        "has_audio": has_audio,
        "file_path": file_path,
    }


def test_compute_layout_keeps_lecturer_first_when_hide_silent_enabled():
    lecturer_key = ("lecturer", False)
    participant_key = ("participant", False)
    streams = {
        lecturer_key: _make_stream(is_admin=True, user_id=1, user_name="Lecturer"),
        participant_key: _make_stream(is_admin=False, user_id=2, user_name="Participant"),
    }
    streams[lecturer_key]["clips"].append(_make_clip(file_path="lecturer.mp4"))
    streams[participant_key]["clips"].append(_make_clip(file_path="participant.mp4"))
    speech_timelines = {
        lecturer_key: [],
        participant_key: [(0.0, 10.0)],
    }

    timeline = compute_layout_timeline(
        streams=streams,
        total_duration=10.0,
        admin_user_id=1,
        hide_silent=True,
        speech_timelines=speech_timelines,
        start_time=0.0,
    )

    assert len(timeline) == 1
    cameras = timeline[0]["cameras"]
    assert [c[0] for c in cameras] == [lecturer_key, participant_key]
    assert {a[0] for a in timeline[0]["audio_sources"]} == {lecturer_key, participant_key}


def test_compute_layout_splits_timeline_on_chat_events():
    share_key = ("share", True)
    streams = {
        share_key: _make_stream(is_admin=False, is_screenshare=True, user_id=100, user_name="Screen"),
    }
    streams[share_key]["clips"].append(_make_clip(file_path="screen.mp4"))
    events = [
        {"time": 2.0, "type": "CHAT", "user": "Alice", "text": "Hello"},
        {"time": 5.0, "type": "Q&A", "user": "Bob", "text": "Question"},
    ]

    timeline = compute_layout_timeline(
        streams=streams,
        total_duration=10.0,
        admin_user_id=None,
        start_time=0.0,
        events=events,
    )

    assert [round(s["start"], 3) for s in timeline] == [0.0, 2.0, 5.0]
    assert [round(s["end"], 3) for s in timeline] == [2.0, 5.0, 10.0]
    assert [s["chat_version"] for s in timeline] == [0, 1, 2]


def test_build_chat_overlay_text_uses_latest_messages_only():
    events = [
        {"time": 1.0, "type": "CHAT", "user": "u1", "text": "m1"},
        {"time": 2.0, "type": "CHAT", "user": "u2", "text": "m2"},
        {"time": 3.0, "type": "Q&A", "user": "u3", "text": "m3"},
    ]

    text = _build_chat_overlay_text(events, t=3.0, max_lines=2)

    assert "[Q&A 00:00:03]" in text
    assert "m3" in text
    assert "[CHAT 00:00:02]" not in text
    assert "m1" not in text


def test_build_chat_overlay_text_wraps_long_lines():
    events = [
        {"time": 12.0, "type": "CHAT", "user": "u1", "text": "this is a very long chat message that should wrap"},
    ]
    text = _build_chat_overlay_text(events, t=12.0, max_lines=6, wrap_chars=12)

    assert "[CHAT 00:00:12]" in text
    assert "this is a very" in text
    assert "long chat" in text


def test_format_elapsed_returns_hh_mm_ss():
    assert _format_elapsed(0.0) == "00:00:00"
    assert _format_elapsed(59.9) == "00:00:59"
    assert _format_elapsed(3661.2) == "01:01:01"


def test_compute_layout_respects_start_time_window():
    camera_key = ("cam", False)
    streams = {camera_key: _make_stream(is_admin=False, user_id=1, user_name="A")}
    streams[camera_key]["clips"].append(_make_clip(start=0.0, duration=120.0, file_path="a.mp4"))

    timeline = compute_layout_timeline(
        streams=streams,
        total_duration=80.0,
        admin_user_id=None,
        start_time=50.0,
    )

    assert len(timeline) == 1
    assert timeline[0]["start"] == 50.0
    assert timeline[0]["end"] == 80.0


def test_parse_chat_and_questions_supports_generic_chat_and_nested_text():
    json_data = {
        "eventLogs": [
            {
                "module": "message.add",
                "relativeTime": 12.3,
                "data": {
                    "type": "chat",
                    "sender": {"displayName": "Alice"},
                    "message": {"text": "Hello from chat"},
                },
            },
            {
                "module": "question.add",
                "relativeTime": 18.0,
                "data": {
                    "author": {"nickname": "Bob"},
                    "payload": {"content": "Question text"},
                },
            },
        ]
    }

    chat, questions = parse_chat_and_questions(json_data)

    assert chat == [
        {"time": 12.3, "user": "Alice", "text": "Hello from chat", "type": "CHAT"}
    ]
    assert questions == [
        {"time": 18.0, "user": "Bob", "text": "Question text", "type": "Q&A"}
    ]
