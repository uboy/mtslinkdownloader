import argparse

from mtslinkdownloader import cli


def test_extract_ids_from_url_regular_record():
    url = "https://my.mts-link.ru/12345678/987654321/record-new/111222333/record-file/444555666"
    event_sessions, record_id = cli.extract_ids_from_url(url)

    assert event_sessions == "111222333"
    assert record_id == "444555666"


def test_extract_ids_from_url_quick_meeting():
    url = "https://my.mts-link.ru/12345678/987654321/record-new/111222333"
    event_sessions, record_id = cli.extract_ids_from_url(url)

    assert event_sessions == "111222333"
    assert record_id is None


def test_extract_ids_from_url_supports_trailing_slash():
    url = "https://my.mts-link.ru/12345678/987654321/record-new/111222333/"
    event_sessions, record_id = cli.extract_ids_from_url(url)

    assert event_sessions == "111222333"
    assert record_id is None


def test_extract_ids_from_url_invalid_domain():
    event_sessions, record_id = cli.extract_ids_from_url(
        "https://example.com/123/456/record-new/111222333"
    )

    assert event_sessions is None
    assert record_id is None


def test_main_returns_error_for_invalid_url(monkeypatch):
    monkeypatch.setattr(
        cli,
        "parse_arguments",
        lambda: argparse.Namespace(
            url="not-a-valid-url",
            session_id=None,
            hide_silent=False,
            max_duration=None,
            start_time=0,
            quality="1080p",
        ),
    )
    monkeypatch.setattr(
        cli,
        "fetch_webinar_data",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    assert cli.main() == 1


def test_main_returns_zero_and_forwards_args(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        cli,
        "parse_arguments",
        lambda: argparse.Namespace(
            url="https://my.mts-link.ru/1/2/record-new/123/record-file/456",
            session_id="sid",
            hide_silent=True,
            max_duration=120.0,
            start_time=30.0,
            quality="720p",
        ),
    )

    def fake_fetch_webinar_data(**kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(cli, "fetch_webinar_data", fake_fetch_webinar_data)

    assert cli.main() == 0
    assert captured == {
        "event_sessions": "123",
        "record_id": "456",
        "session_id": "sid",
        "hide_silent": True,
        "max_duration": 120.0,
        "start_time": 30.0,
        "quality": "720p",
    }


def test_main_returns_error_when_fetch_fails(monkeypatch):
    monkeypatch.setattr(
        cli,
        "parse_arguments",
        lambda: argparse.Namespace(
            url="https://my.mts-link.ru/1/2/record-new/123",
            session_id=None,
            hide_silent=False,
            max_duration=None,
            start_time=0,
            quality="1080p",
        ),
    )
    monkeypatch.setattr(cli, "fetch_webinar_data", lambda **_: None)

    assert cli.main() == 1
