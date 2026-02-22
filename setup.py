from setuptools import setup, find_packages

setup(
    name='mtslinkdownloader',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'httpx>=0.27.2',
        'moviepy>=1.0.3',
        'tqdm>=4.66.6',
    ],
    entry_points={
        'console_scripts': [
            'mtslinkdownloader=mtslinkdownloader.cli:main',
        ],
    },
    python_requires=">=3.9",
)
