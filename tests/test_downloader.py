import pytest

from mtslinkdownloader.downloader import construct_json_data_url


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
