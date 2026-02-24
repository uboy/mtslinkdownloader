import argparse
import logging
import re
from typing import Optional, Tuple

from .utils import initialize_logger
from .webinar import fetch_webinar_data

URL_PATTERN = re.compile(
    r'^https://[^/]+\.mts-link\.ru/(?:[^/]+/)*\d+/\d+/record-new/(\d+)(?:/record-file/(\d+))?/?$'
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='mtslinkdownloader - tool for downloading MTS Link webinars.'
    )
    parser.add_argument(
        'url',
        help=(
            'Webinar link in one of the following formats: '
            'https://my.mts-link.ru/12345678/987654321/record-new/123456789/record-file/1234567890 or '
            'https://my.mts-link.ru/12345678/987654321/record-new/123456789'
        )
    )
    parser.add_argument(
        '--session-id',
        help='[Optional] sessionId token for accessing private recordings.'
    )
    parser.add_argument(
        '--hide-silent',
        action='store_true',
        default=False,
        help='Hide cameras of non-speaking participants.'
    )
    parser.add_argument(
        '--max-duration',
        type=float,
        default=None,
        help='Limit output video to N seconds from start-time (e.g. 180 for 3 minutes).'
    )
    parser.add_argument(
        '--start-time',
        type=float,
        default=0,
        help='Start processing from this time offset in seconds (e.g. 2400).'
    )
    parser.add_argument(
        '--quality',
        choices=['720p', '1080p'],
        default='1080p',
        help='Output video quality (resolution). Default is 1080p.'
    )
    return parser.parse_args()


def extract_ids_from_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    match = URL_PATTERN.match(url.strip())

    if match:
        return match.group(1), match.group(2)

    return None, None


def main():
    initialize_logger(force=False)
    # Silence httpx logs to keep progress bars readable
    logging.getLogger("httpx").setLevel(logging.WARNING)

    args = parse_arguments()

    event_sessions, record_id = extract_ids_from_url(args.url)
    if event_sessions is None:
        logging.error('Invalid URL format. Please check the link.')
        return 1

    logging.info(f'Starting download: event_sessions={event_sessions}, record_id={record_id}')
    try:
        if fetch_webinar_data(
            event_sessions=event_sessions,
            record_id=record_id,
            session_id=args.session_id,
            hide_silent=args.hide_silent,
            max_duration=args.max_duration,
            start_time=args.start_time,
            quality=args.quality
        ):
            logging.info('Download completed.')
            return 0
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user. Exiting safely...")
        return 130
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return 1
    finally:
        from .utils import restore_terminal
        restore_terminal()

    return 1

if __name__ == '__main__':
    raise SystemExit(main())
