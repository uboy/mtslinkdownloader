import subprocess
from types import SimpleNamespace

import pytest

from mtslinkdownloader import processor
from mtslinkdownloader.processor import (
    _log_source_switch_diagnostics,
    compute_layout_timeline,
    _build_chat_overlay_text,
    _format_elapsed,
    _resolve_total_duration,
    parse_chat_and_questions,
    parse_presentation_timeline,
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

    assert "[Q&A 00:00:03] u3" in text
    assert "m3" in text
    assert "[CHAT 00:00:02]" not in text
    assert "m1" not in text


def test_build_chat_overlay_text_wraps_long_lines():
    events = [
        {"time": 12.0, "type": "CHAT", "user": "u1", "text": "this is a very long chat message that should wrap"},
    ]
    text = _build_chat_overlay_text(events, t=12.0, max_lines=6, wrap_chars=12)

    assert "[CHAT 00:00:12] u1" in text
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


def test_compute_layout_uses_speech_overlap_not_midpoint_only():
    lecturer_key = ("lecturer", False)
    speaker_key = ("speaker", False)
    streams = {
        lecturer_key: _make_stream(is_admin=True, user_id=1, user_name="Lecturer"),
        speaker_key: _make_stream(is_admin=False, user_id=2, user_name="Speaker"),
    }
    streams[lecturer_key]["clips"].append(_make_clip(start=0.0, duration=10.0, file_path="lecturer.mp4"))
    streams[speaker_key]["clips"].append(_make_clip(start=0.0, duration=10.0, file_path="speaker.mp4"))
    speech_timelines = {
        lecturer_key: [],
        speaker_key: [(0.0, 0.3)],
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
    assert [c[0] for c in timeline[0]["cameras"]] == [lecturer_key, speaker_key]


def test_compute_layout_places_first_speaking_participant_next_to_lecturer():
    lecturer_key = ("lecturer", False)
    alpha_key = ("alpha", False)
    zeta_key = ("zeta", False)
    streams = {
        lecturer_key: _make_stream(is_admin=True, user_id=1, user_name="Lecturer"),
        alpha_key: _make_stream(is_admin=False, user_id=2, user_name="Alpha"),
        zeta_key: _make_stream(is_admin=False, user_id=3, user_name="Zeta"),
    }
    for key, path in ((lecturer_key, "lecturer.mp4"), (alpha_key, "alpha.mp4"), (zeta_key, "zeta.mp4")):
        streams[key]["clips"].append(_make_clip(start=0.0, duration=10.0, file_path=path))
    speech_timelines = {
        lecturer_key: [],
        alpha_key: [(0.0, 2.0)],
        zeta_key: [(0.0, 8.0)],
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
    assert [c[0] for c in timeline[0]["cameras"]] == [lecturer_key, zeta_key, alpha_key]


def test_compute_layout_hide_silent_keeps_audio_for_hidden_participants():
    lecturer_key = ("lecturer", False)
    participant_key = ("participant", False)
    streams = {
        lecturer_key: _make_stream(is_admin=True, user_id=1, user_name="Lecturer"),
        participant_key: _make_stream(is_admin=False, user_id=2, user_name="Participant"),
    }
    streams[lecturer_key]["clips"].append(_make_clip(start=0.0, duration=10.0, file_path="lecturer.mp4"))
    streams[participant_key]["clips"].append(_make_clip(start=0.0, duration=10.0, file_path="participant.mp4"))
    speech_timelines = {
        lecturer_key: [],
        participant_key: [],
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
    assert [c[0] for c in timeline[0]["cameras"]] == [lecturer_key]
    assert {a[0] for a in timeline[0]["audio_sources"]} == {lecturer_key, participant_key}


def test_resolve_total_duration_applies_max_duration_window():
    duration = _resolve_total_duration(
        json_data={"duration": 120.0},
        max_duration=15.0,
        start_time=40.0,
    )

    assert duration == 55.0


def test_parse_presentation_timeline_uses_slide_index_changes():
    json_data = {
        "duration": 30.0,
        "eventLogs": [
            {
                "module": "presentation.update",
                "relativeTime": 0.0,
                "data": {
                    "isActive": True,
                    "slideIndex": 0,
                    "fileReference": {
                        "file": {
                            "slides": [
                                {"url": "s1.jpg"},
                                {"url": "s2.jpg"},
                            ]
                        }
                    },
                },
            },
            {
                "module": "presentation.update",
                "relativeTime": 5.0,
                "data": {"isActive": True, "slideIndex": 0},
            },
            {
                "module": "presentation.update",
                "relativeTime": 10.0,
                "data": {"isActive": True, "slideIndex": 1},
            },
            {
                "module": "presentation.update",
                "relativeTime": 20.0,
                "data": {"isActive": False},
            },
        ],
    }

    slides, timeline = parse_presentation_timeline(json_data)

    assert slides == ["s1.jpg", "s2.jpg"]
    assert timeline == [(0.0, 10.0, "s1.jpg"), (10.0, 20.0, "s2.jpg")]


def test_parse_presentation_timeline_keeps_urls_from_multiple_file_references():
    json_data = {
        "duration": 40.0,
        "eventLogs": [
            {
                "module": "presentation.update",
                "relativeTime": 0.0,
                "data": {
                    "isActive": True,
                    "slideIndex": 0,
                    "fileReference": {
                        "file": {
                            "slides": [
                                {"url": "a1.jpg"},
                                {"url": "a2.jpg"},
                            ]
                        }
                    },
                },
            },
            {
                "module": "presentation.update",
                "relativeTime": 10.0,
                "data": {"isActive": True, "slideIndex": 1},
            },
            {
                "module": "presentation.update",
                "relativeTime": 20.0,
                "data": {
                    "isActive": True,
                    "slideIndex": 0,
                    "fileReference": {
                        "file": {
                            "slides": [
                                {"url": "b1.jpg"},
                                {"url": "b2.jpg"},
                            ]
                        }
                    },
                },
            },
            {
                "module": "presentation.update",
                "relativeTime": 30.0,
                "data": {"isActive": False},
            },
        ],
    }

    slides, timeline = parse_presentation_timeline(json_data)

    assert timeline == [
        (0.0, 10.0, "a1.jpg"),
        (10.0, 20.0, "a2.jpg"),
        (20.0, 30.0, "b1.jpg"),
    ]
    assert slides == ["a1.jpg", "a2.jpg", "b1.jpg", "b2.jpg"]


def test_parse_presentation_timeline_switches_deck_on_direct_url_before_slides_list_update():
    json_data = {
        "duration": 30.0,
        "eventLogs": [
            {
                "module": "presentation.update",
                "relativeTime": 0.0,
                "data": {
                    "isActive": True,
                    "slideIndex": 0,
                    "fileReference": {
                        "id": "deck-a",
                        "file": {
                            "slides": [
                                {"url": "a1.jpg"},
                                {"url": "a2.jpg"},
                            ]
                        },
                    },
                },
            },
            {
                "module": "presentation.update",
                "relativeTime": 10.0,
                "data": {
                    "isActive": True,
                    "currentSlideUrl": "b1.jpg",
                },
            },
            {
                "module": "presentation.update",
                "relativeTime": 15.0,
                "data": {
                    "isActive": True,
                    "slideIndex": 0,
                    "fileReference": {
                        "id": "deck-b",
                        "file": {
                            "slides": [
                                {"url": "b1.jpg"},
                                {"url": "b2.jpg"},
                            ]
                        },
                    },
                },
            },
            {
                "module": "presentation.update",
                "relativeTime": 20.0,
                "data": {
                    "isActive": False,
                },
            },
        ],
    }

    slides, timeline = parse_presentation_timeline(json_data)

    assert timeline == [
        (0.0, 10.0, "a1.jpg"),
        (10.0, 20.0, "b1.jpg"),
    ]
    assert slides == ["a1.jpg", "b1.jpg", "a2.jpg", "b2.jpg"]


def test_source_switch_diagnostics_warns_when_presentation_and_screenshare_overlap(caplog):
    streams = {
        ("screen", True): {
            "is_screenshare": True,
            "clips": [
                {"relative_time": 0.0, "duration": 120.0},
            ],
        }
    }
    json_data = {
        "eventLogs": [
            {"module": "presentation.update", "relativeTime": 0.0, "data": {"isActive": True}},
            {"module": "presentation.update", "relativeTime": 90.0, "data": {"isActive": False}},
        ]
    }
    slide_timeline = [(0.0, 90.0, "slide.jpg")]

    with caplog.at_level("WARNING"):
        _log_source_switch_diagnostics(
            json_data=json_data,
            streams=streams,
            raw_slide_timeline=slide_timeline,
            start_time=0.0,
            total_duration=120.0,
        )

    assert any("both active" in message for message in caplog.messages)


def test_run_ffmpeg_treats_signal_2_as_interrupted(monkeypatch):
    monkeypatch.setattr(processor, "STOP_REQUESTED", False)
    monkeypatch.setattr(processor, "_get_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=255,
            stderr="Exiting normally, received signal 2.",
        ),
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        processor._run_ffmpeg(["-version"], desc="probe")
