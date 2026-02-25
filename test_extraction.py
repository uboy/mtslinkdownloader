import json
import sys
import os
sys.path.append(os.getcwd())
from mtslinkdownloader.processor import _extract_slide_url

def main():
    # Sample data from the dump
    data = {
        'id': 55691953,
        'isActive': True,
        'url': 'https://example.com/deck.pptx',
        'slide': {
            'id': 1726250395,
            'url': 'https://example.com/slide10.jpg'
        }
    }
    
    res = _extract_slide_url(data, [], None)
    print(f"Extracted URL: {res}")

if __name__ == "__main__":
    main()
