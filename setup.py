from setuptools import setup, find_packages

setup(
    name='mtslinkdownloader',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'httpx>=0.27.2',
        'imageio_ffmpeg>=0.5.1',
        'tqdm>=4.66.6',
    ],
    extras_require={
        'gui': [
            'customtkinter>=5.2.2',
            'darkdetect>=0.8.0',
        ],
        'build': [
            'pyinstaller>=6.11.1',
        ],
    },
    entry_points={
        'console_scripts': [
            'mtslinkdownloader=mtslinkdownloader.cli:main',
        ],
    },
    python_requires=">=3.9",
)
