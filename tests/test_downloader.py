import pytest

from mtslinkdownloader.downloader import construct_json_data_url, download_video_chunk


def test_construct_json_data_url_for_regular_record():
    url = construct_json_data_url("111222333", "444555666")

    assert (
        url
        == "https://my.mts-link.ru/api/event-sessions/111222333/record-files/444555666/flow?withoutCuts=false"
    )


@pytest.mark.parametrize("recording_id", [None, ""])
def test_construct_json_data_url_for_quick_meeting(recording_id):
    url = construct_json_data_url("111222333", recording_id)

    assert (
        url
        == "https://my.mts-link.ru/api/eventsessions/111222333/record?withoutCuts=false"
    )


@pytest.mark.parametrize("event_session_id", [None, ""])
def test_construct_json_data_url_raises_for_missing_event_session(event_session_id):
    with pytest.raises(ValueError, match="Missing webinar event session ID"):
        construct_json_data_url(event_session_id, "444555666")


class _DummyResponse:
    def __init__(self, headers=None, chunks=None):
        self.headers = headers or {}
        self._chunks = chunks or []

    def raise_for_status(self):
        return None

    def iter_bytes(self, chunk_size=262144):
        for chunk in self._chunks:
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_video_chunk_skips_download_when_size_matches(tmp_path):
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"abcd")

    class _Client:
        def __init__(self):
            self.stream_calls = 0

        def head(self, _url):
            return _DummyResponse(headers={"content-length": "4"})

        def stream(self, method, _url):
            assert method == "GET"
            self.stream_calls += 1
            return _DummyResponse(chunks=[b"new"])

    client = _Client()
    path = download_video_chunk("https://example.com/clip.mp4", str(tmp_path), client=client)

    assert path == str(target)
    assert target.read_bytes() == b"abcd"
    assert client.stream_calls == 0


def test_download_video_chunk_redownloads_when_head_check_fails(tmp_path):
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"old")

    class _Client:
        def __init__(self):
            self.stream_calls = 0

        def head(self, _url):
            raise RuntimeError("HEAD disabled")

        def stream(self, method, _url):
            assert method == "GET"
            self.stream_calls += 1
            return _DummyResponse(chunks=[b"new-content"])

    client = _Client()
    path = download_video_chunk("https://example.com/clip.mp4", str(tmp_path), client=client)

    assert path == str(target)
    assert target.read_bytes() == b"new-content"
    assert client.stream_calls == 1
