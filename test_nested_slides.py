import sys
import os
sys.path.append(os.getcwd())
from mtslinkdownloader.processor import parse_presentation_timeline

def main():
    # Sample data with nested slides list
    json_data = {
        'duration': 1000,
        'eventLogs': [
            {
                'module': 'presentation.update',
                'relativeTime': 10,
                'data': {
                    'isActive': True,
                    'fileReference': {
                        'file': {
                            'id': 1,
                            'slides': [
                                [{'url': 's1.jpg'}, {'url': 's2.jpg'}]
                            ]
                        }
                    },
                    'slide': {'url': 's1.jpg'}
                }
            }
        ]
    }
    
    slides, timeline = parse_presentation_timeline(json_data)
    print(f"Slides: {slides}")
    print(f"Timeline: {timeline}")

if __name__ == "__main__":
    main()
